#!/usr/bin/env python3
"""
Real LIBERO Trajectory Code Analyzer - Uses actual model inference
"""

import argparse
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from torchvision import transforms
from einops import rearrange

# Add the project root to the path
sys.path.append('/ssd2/UniVLA')

from latent_action_model.genie.modules.lam import UncontrolledDINOLatentActionModel, ControllableDINOLatentActionModel

def load_latent_action_model(checkpoint_path: str, device: str = "cuda"):
    """Load the trained latent action model from checkpoint."""
    
    # Model configuration (from config files)
    model_config = {
        'in_dim': 3,
        'model_dim': 768,
        'latent_dim': 128,
        'num_latents': 16,
        'patch_size': 14,
        'enc_blocks': 12,
        'dec_blocks': 12,
        'num_heads': 12,
        'dropout': 0.0
    }
    
    # Check if it's stage-1 or stage-2 based on checkpoint path
    if 'stage1' in checkpoint_path:
        model = UncontrolledDINOLatentActionModel(**model_config)
    else:
        model = ControllableDINOLatentActionModel(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict (remove 'lam.' prefix if exists)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('lam.'):
            new_key = key[4:]  # Remove 'lam.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Load state dict
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device).eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    return model

def load_libero_trajectory(data_root: str, task_name: str, episode_idx: int = 0) -> Dict:
    """Load a specific LIBERO trajectory from the dataset."""
    
    # Construct dataset path
    dataset_path = os.path.join(data_root, f"{task_name}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    try:
        # Load RLDS dataset
        dataset = tfds.builder_from_directory(dataset_path)
        dataset = dataset.as_dataset(split='train')
        
        # Get specific episode
        episode = list(dataset.take(episode_idx + 1))[-1]
        
        print(f"✓ Loaded LIBERO trajectory: {task_name}, episode {episode_idx}")
        print(f"Episode keys: {list(episode.keys())}")
        
        return episode
        
    except Exception as e:
        print(f"❌ Error loading trajectory: {e}")
        return None

def process_libero_images(episode: Dict, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Process LIBERO episode images into tensor format."""
    
    # Debug: print episode structure
    print(f"Episode keys: {list(episode.keys())}")
    
    # Get images from episode - try different possible structures
    images = None
    if 'observation' in episode and 'image' in episode['observation']:
        images = episode['observation']['image']
    elif 'steps' in episode:
        # Convert TF dataset to list
        steps_list = list(episode['steps'])
        print(f"Number of steps: {len(steps_list)}")
        if len(steps_list) > 0:
            print(f"First step keys: {list(steps_list[0].keys())}")
            
            # Try to get images from steps
            if 'observation' in steps_list[0] and 'image' in steps_list[0]['observation']:
                images = [step['observation']['image'] for step in steps_list]
            elif 'image' in steps_list[0]:
                images = [step['image'] for step in steps_list]
    
    if images is None:
        raise KeyError("Could not find images in episode. Available keys: " + str(list(episode.keys())))
    
    # Convert to numpy and process
    processed_images = []
    
    for i in range(len(images)):
        # Convert TF tensor to numpy
        img_np = images[i].numpy()
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_np)
        
        # Resize to target size
        img_pil = img_pil.resize(target_size)
        
        # Convert to tensor and normalize
        img_tensor = transforms.ToTensor()(img_pil)  # (C, H, W)
        
        processed_images.append(img_tensor)
    
    # Stack into sequence tensor
    video_sequence = torch.stack(processed_images)  # (T, C, H, W)
    
    print(f"Processed {len(processed_images)} images into tensor of shape: {video_sequence.shape}")
    
    return video_sequence

def extract_real_codes_from_trajectory(
    model,
    video_sequence: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Extract real codes from trajectory using actual model inference."""
    
    print("Extracting real latent action codes...")
    
    # Add batch dimension
    video_sequence = video_sequence.unsqueeze(0)  # (1, T, C, H, W)
    B, T, C, H, W = video_sequence.shape
    
    # Move to device
    video_sequence = video_sequence.to(device)
    
    print(f"Processing video sequence with shape: {video_sequence.shape}")
    
    # Check if this is Stage-2
    is_stage2 = hasattr(model, 'vq_action') and hasattr(model, 'vq')
    
    if is_stage2:
        print(f"Processing {T} timesteps with Stage-2 model...")
        print("Using actual model inference for Stage-2...")
        
        # Process in pairs (T=2) as required by the model
        all_codes = []
        all_codes_uncontrol = []
        all_z_q = []
        all_z_q_uncontrol = []
        
        # Process using sliding window approach (like training)
        # Use sliding window: 0-15, 1-16, 2-17, etc.
        for t in range(0, T - 15):  # Sliding window: each timestep
            if t + 15 < T:
                # Get first and last frame from 16-frame sliding window
                first_frame = video_sequence[:, t, :, :, :].unsqueeze(1)  # (B, 1, C, H, W)
                last_frame = video_sequence[:, t + 15, :, :, :].unsqueeze(1)  # (B, 1, C, H, W)
                pair_frames = torch.cat([first_frame, last_frame], dim=1)  # (B, 2, C, H, W)
                
                # Use actual model inference
                with torch.no_grad():
                    model_outputs = model.vq_encode(pair_frames)
                
                # Extract codes for this timestep
                indices = model_outputs["indices"]  # (1, 4)
                indices_uncontrol = model_outputs["indices_uncontrol"]  # (1, 4)
                
                # Add batch dimension to make it (1, 1, 4)
                all_codes.append(indices.unsqueeze(1))  # (1, 1, 4)
                all_codes_uncontrol.append(indices_uncontrol.unsqueeze(1))  # (1, 1, 4)
                all_z_q.append(model_outputs["z_q"])  # (1, 1, 4, 128)
                all_z_q_uncontrol.append(model_outputs["z_q_uncontrol"])  # (1, 1, 4, 128)
            else:
                # Handle last frame
                break
        
        # Stack all codes
        indices = torch.cat(all_codes, dim=1)  # (B, T-1, num_codes)
        z_q = torch.cat(all_z_q, dim=1)  # (B, T-1, num_codes, latent_dim)
        indices_uncontrol = torch.cat(all_codes_uncontrol, dim=1)  # (B, T-1, num_codes)
        z_q_uncontrol = torch.cat(all_z_q_uncontrol, dim=1)  # (B, T-1, num_codes, latent_dim)
        
        print("✓ Successfully used actual Stage-2 model inference!")
        
        # Create outputs dictionary for Stage-2
        outputs = {
            "indices": indices,
            "z_q": z_q,
            "indices_uncontrol": indices_uncontrol,
            "z_q_uncontrol": z_q_uncontrol,
            "patches": None,
            "z": None,
            "emb": None,
            "z_uncontrol": None,
            "emb_uncontrol": None
        }
        
    else:
        print(f"Processing {T} timesteps with Stage-1 model...")
        print("Using actual model inference for Stage-1...")
        
        # For Stage-1, process using first and last frame approach (like training)
        all_codes = []
        all_z_q = []
        
        # Process using sliding window approach (like training)
        # Use sliding window: 0-15, 1-16, 2-17, etc.
        for t in range(0, T - 15):  # Sliding window: each timestep
            if t + 15 < T:
                # Get first and last frame from 16-frame sliding window
                first_frame = video_sequence[:, t, :, :, :].unsqueeze(1)  # (B, 1, C, H, W)
                last_frame = video_sequence[:, t + 15, :, :, :].unsqueeze(1)  # (B, 1, C, H, W)
                pair_frames = torch.cat([first_frame, last_frame], dim=1)  # (B, 2, C, H, W)
                
                with torch.no_grad():
                    model_outputs = model.vq_encode(pair_frames)
                
                # Extract codes for this timestep
                indices = model_outputs["indices"]  # (1, 4)
                
                # Add batch dimension to make it (1, 1, 4)
                all_codes.append(indices.unsqueeze(1))  # (1, 1, 4)
                all_z_q.append(model_outputs["z_q"])  # (1, 1, 4, 128)
            else:
                # Handle last frame
                break
        
        # Stack all codes
        indices = torch.cat(all_codes, dim=1)
        z_q = torch.cat(all_z_q, dim=1)
        
        print("✓ Successfully used actual Stage-1 model inference!")
        
        # Create outputs dictionary for Stage-1
        outputs = {
            "indices": indices,
            "z_q": z_q,
            "patches": None,
            "z": None,
            "emb": None
        }
    
    print(f"✓ Extracted real codes with keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    return outputs

def analyze_code_quality(outputs: Dict[str, torch.Tensor]) -> None:
    """Analyze the quality of extracted codes."""
    
    print("\n" + "="*60)
    print("REAL CODE QUALITY ANALYSIS")
    print("="*60)
    
    # Analyze controlled codes
    indices = outputs["indices"]
    all_codes = indices.cpu().numpy().flatten()
    unique_codes = np.unique(all_codes)
    
    print(f"\nControlled Codes Analysis:")
    print(f"Total codes: {len(all_codes)}")
    print(f"Unique codes used: {len(unique_codes)}")
    print(f"Code usage rate: {len(unique_codes)/4:.3f} ({len(unique_codes)}/4)")
    print(f"Code distribution: {np.bincount(all_codes, minlength=4)}")
    
    # Analyze uncontrolled codes if available
    if "indices_uncontrol" in outputs:
        indices_uncontrol = outputs["indices_uncontrol"]
        all_codes_uncontrol = indices_uncontrol.cpu().numpy().flatten()
        unique_codes_uncontrol = np.unique(all_codes_uncontrol)
        
        print(f"\nUncontrolled Codes Analysis:")
        print(f"Total codes: {len(all_codes_uncontrol)}")
        print(f"Unique codes used: {len(unique_codes_uncontrol)}")
        print(f"Code usage rate: {len(unique_codes_uncontrol)/4:.3f} ({len(unique_codes_uncontrol)}/4)")
        print(f"Code distribution: {np.bincount(all_codes_uncontrol, minlength=4)}")
        
        # Compare controlled vs uncontrolled
        print(f"\nComparison:")
        print(f"Controlled code diversity: {len(unique_codes)}/4")
        print(f"Uncontrolled code diversity: {len(unique_codes_uncontrol)}/4")
        
        # Check for code overlap
        overlap = len(set(unique_codes) & set(unique_codes_uncontrol))
        print(f"Code overlap: {overlap} codes")
    
    # Analyze temporal patterns
    print(f"\nTemporal Analysis:")
    print(f"Code changes per timestep:")
    for t in range(min(10, indices.shape[1])):  # Show first 10 timesteps
        if t > 0:
            prev_codes = indices[0, t-1, :].cpu().numpy()
            curr_codes = indices[0, t, :].cpu().numpy()
            changes = np.sum(prev_codes != curr_codes)
            print(f"  Timestep {t}: {changes} changes")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    controlled_usage = len(unique_codes) / 16
    if controlled_usage > 0.5:
        print("✓ Good controlled code diversity")
    elif controlled_usage > 0.25:
        print("⚠️  Moderate controlled code diversity")
    else:
        print("❌ Low controlled code diversity")
    
    if "indices_uncontrol" in outputs:
        uncontrolled_usage = len(unique_codes_uncontrol) / 16
        if uncontrolled_usage > 0.5:
            print("✓ Good uncontrolled code diversity")
        elif uncontrolled_usage > 0.25:
            print("⚠️  Moderate uncontrolled code diversity")
        else:
            print("❌ Low uncontrolled code diversity")

def create_code_visualization_frame(
    image: np.ndarray, 
    codes: np.ndarray, 
    timestep: int,
    codebook_size: int = 16,
    codes_uncontrol: np.ndarray = None,
    is_stage2: bool = False
) -> np.ndarray:
    """Create a visualization frame showing the image and corresponding codes."""
    
    # Convert image to PIL for easier text overlay
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load smaller fonts for better readability
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8)
    except:
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    # Add timestep information
    timestep_text = f"Timestep: {timestep}"
    draw.text((11, 11), timestep_text, fill=(0, 0, 0), font=font_small)  # Shadow
    draw.text((10, 10), timestep_text, fill=(255, 255, 255), font=font_small)  # Main text
    
    # Add codes information
    if is_stage2 and codes_uncontrol is not None:
        codes_text = f"Controlled: {codes[:4].tolist()}"  # Show all 4 codes
        draw.text((11, 31), codes_text, fill=(0, 0, 0), font=font_small)  # Shadow
        draw.text((10, 30), codes_text, fill=(255, 255, 255), font=font_small)  # Main text
        
        codes_uncontrol_text = f"Uncontrolled: {codes_uncontrol[:4].tolist()}"  # Show all 4 codes
        draw.text((11, 51), codes_uncontrol_text, fill=(0, 0, 0), font=font_small)  # Shadow
        draw.text((10, 50), codes_uncontrol_text, fill=(255, 255, 0), font=font_small)  # Main text (yellow)
    else:
        codes_text = f"Codes: {codes[:4].tolist()}"  # Show all 4 codes
        draw.text((11, 31), codes_text, fill=(0, 0, 0), font=font_small)  # Shadow
        draw.text((10, 30), codes_text, fill=(255, 255, 255), font=font_small)  # Main text
    
    # Create code visualization bar(s)
    if is_stage2 and codes_uncontrol is not None:
        # Two bars: controlled (top) and uncontrolled (bottom)
        code_bar_height = 20
        controlled_bar_y = pil_image.height - 2 * code_bar_height - 10
        uncontrolled_bar_y = pil_image.height - code_bar_height - 5
        
        # Draw controlled codes bar (blue theme)
        draw.rectangle([5, controlled_bar_y, pil_image.width - 5, controlled_bar_y + code_bar_height], 
                       fill=(0, 0, 100))  # Dark blue background
        
        # Draw uncontrolled codes bar (red theme)
        draw.rectangle([5, uncontrolled_bar_y, pil_image.width - 5, uncontrolled_bar_y + code_bar_height], 
                       fill=(100, 0, 0))  # Dark red background
        
        # Draw controlled codes
        num_codes = len(codes)
        block_width = (pil_image.width - 10) // num_codes
        
        for i, code in enumerate(codes):
            x1 = 5 + i * block_width
            x2 = 5 + (i + 1) * block_width
            
            # Blue color scheme for controlled codes
            hue = (code * 137) % 180
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue/180.0, 0.8, 0.9)
            color = tuple(int(c * 255) for c in rgb)
            # Make it more blue-tinted
            color = (min(255, color[0] + 50), color[1], min(255, color[2] + 50))
            
            draw.rectangle([x1, controlled_bar_y, x2, controlled_bar_y + code_bar_height], fill=color)
            
            if block_width > 12:
                draw.text((x1 + 2, controlled_bar_y + 2), str(code), fill=(255, 255, 255), font=font_tiny)
        
        # Draw uncontrolled codes
        for i, code in enumerate(codes_uncontrol):
            x1 = 5 + i * block_width
            x2 = 5 + (i + 1) * block_width
            
            # Red color scheme for uncontrolled codes
            hue = (code * 137) % 180
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue/180.0, 0.8, 0.9)
            color = tuple(int(c * 255) for c in rgb)
            # Make it more red-tinted
            color = (min(255, color[0] + 50), color[1], color[2])
            
            draw.rectangle([x1, uncontrolled_bar_y, x2, uncontrolled_bar_y + code_bar_height], fill=color)
            
            if block_width > 12:
                draw.text((x1 + 2, uncontrolled_bar_y + 2), str(code), fill=(255, 255, 255), font=font_tiny)
        
        # Add labels
        draw.text((5, controlled_bar_y - 15), "Controlled Codes", fill=(255, 255, 255), font=font_tiny)
        draw.text((5, uncontrolled_bar_y - 15), "Uncontrolled Codes", fill=(255, 255, 0), font=font_tiny)
        
    else:
        # Single bar for Stage-1 (original behavior)
        code_bar_height = 25
        code_bar_y = pil_image.height - code_bar_height - 5
        
        # Draw code bar background
        draw.rectangle([5, code_bar_y, pil_image.width - 5, code_bar_y + code_bar_height], 
                       fill=(0, 0, 0))
        
        # Draw individual code blocks
        num_codes = len(codes)
        block_width = (pil_image.width - 10) // num_codes
        
        for i, code in enumerate(codes):
            x1 = 5 + i * block_width
            x2 = 5 + (i + 1) * block_width
            
            # Better color distribution
            hue = (code * 137) % 180  # Better color distribution
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue/180.0, 0.8, 0.9)
            color = tuple(int(c * 255) for c in rgb)
            
            draw.rectangle([x1, code_bar_y, x2, code_bar_y + code_bar_height], fill=color)
            
            # Add code number only if block is wide enough
            if block_width > 12:
                draw.text((x1 + 2, code_bar_y + 2), str(code), fill=(255, 255, 255), font=font_tiny)
    
    return np.array(pil_image)

def create_codes_video(
    video_sequence: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    save_path: str,
    fps: int = 10
) -> None:
    """Create a video showing the trajectory with latent action codes overlay."""
    
    # Get codes
    indices = outputs["indices"]  # (B, T-1, num_codes)
    B, T, num_codes = indices.shape
    
    # Check if this is Stage-2 (has uncontrolled codes)
    is_stage2 = "indices_uncontrol" in outputs
    if is_stage2:
        indices_uncontrol = outputs["indices_uncontrol"]
        print("Creating Stage-2 video with controlled and uncontrolled codes...")
    else:
        print("Creating Stage-1 video with controlled codes...")
    
    # Convert video sequence to numpy
    if video_sequence.dim() == 5:  # (B, T, C, H, W)
        video_sequence = video_sequence[0]  # Remove batch dimension
    
    video_np = video_sequence.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
    
    # Normalize to [0, 1] if needed
    if video_np.max() > 1.0:
        video_np = video_np / 255.0
    
    # Create video writer
    height, width = video_np.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    print(f"Creating video with {T} frames...")
    
    # Process each frame
    for t in range(T):
        if t < T:  # We have T-1 codes for T frames
            codes = indices[0, t, :].cpu().numpy()
            codes_uncontrol = indices_uncontrol[0, t, :].cpu().numpy() if is_stage2 else None
        else:
            codes = np.zeros(num_codes)  # Last frame
            codes_uncontrol = np.zeros(num_codes) if is_stage2 else None
        
        # Create visualization frame
        frame = create_code_visualization_frame(
            video_np[t], 
            codes, 
            t, 
            codes_uncontrol=codes_uncontrol,
            is_stage2=is_stage2
        )
        
        # Convert BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(frame_bgr)
        
        if t % 5 == 0:
            print(f"Processed frame {t}/{T}")
    
    # Release video writer
    out.release()
    print(f"✓ Saved codes visualization video to {save_path}")

def create_code_sequence_plot(
    outputs: Dict[str, torch.Tensor],
    save_path: str
) -> None:
    """Create a plot showing the code sequence over time."""
    
    indices = outputs["indices"]  # (B, T-1, num_codes)
    B, T, num_codes = indices.shape
    
    # Check if this is Stage-2 (has uncontrolled codes)
    is_stage2 = "indices_uncontrol" in outputs
    if is_stage2:
        indices_uncontrol = outputs["indices_uncontrol"]
        print("Creating Stage-2 plot with controlled and uncontrolled codes...")
    else:
        print("Creating Stage-1 plot with controlled codes...")
    
    # Convert to numpy
    codes_np = indices[0].cpu().numpy()  # (T-1, num_codes)
    
    if is_stage2:
        # Create figure with single column for Stage-2 (like Stage-1)
        # Total: 4 controlled + 4 uncontrolled = 8 subplots
        total_codes = num_codes * 2  # 4 + 4 = 8
        fig, axes = plt.subplots(total_codes, 1, figsize=(12, 2 * total_codes))
        if total_codes == 1:
            axes = [axes]
        
        # Plot controlled codes (first 4 subplots)
        codes_uncontrol_np = indices_uncontrol[0].cpu().numpy()  # (T-1, num_codes)
        
        for i in range(num_codes):
            axes[i].plot(codes_np[:, i], 'o-', linewidth=2, markersize=4, color='blue', alpha=0.7)
            axes[i].set_title(f'Controlled Code {i}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Code Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-0.5, 15.5)
        
        # Plot uncontrolled codes (next 4 subplots)
        for i in range(num_codes):
            axes[i + num_codes].plot(codes_uncontrol_np[:, i], 'o-', linewidth=2, markersize=4, color='red', alpha=0.7)
            axes[i + num_codes].set_title(f'Uncontrolled Code {i}', fontsize=10, fontweight='bold')
            axes[i + num_codes].set_xlabel('Timestep')
            axes[i + num_codes].set_ylabel('Code Value')
            axes[i + num_codes].grid(True, alpha=0.3)
            axes[i + num_codes].set_ylim(-0.5, 15.5)
        
    else:
        # Create figure for Stage-1 (original behavior)
        fig, axes = plt.subplots(num_codes, 1, figsize=(12, 2 * num_codes))
        if num_codes == 1:
            axes = [axes]
        
        # Plot each code position over time
        for i in range(num_codes):
            axes[i].plot(codes_np[:, i], 'o-', linewidth=2, markersize=4)
            axes[i].set_title(f'Code Position {i}')
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Code Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-0.5, 15.5)  # Assuming codebook size of 16
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved code sequence plot to {save_path}")

def save_codes_to_file(outputs: Dict[str, torch.Tensor], save_path: str) -> None:
    """Save extracted codes to file."""
    
    # Convert to CPU and numpy for saving
    save_data = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            save_data[key] = value.cpu().numpy()
        else:
            save_data[key] = value
    
    # Save as numpy file
    np.savez(save_path, **save_data)
    print(f"✓ Saved codes to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract real latent action codes from LIBERO trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to latent action model checkpoint")
    parser.add_argument("--data_root", type=str, default="/ssd1/openpi_official/datasets/libero_raw",
                       help="Path to LIBERO dataset root")
    parser.add_argument("--task_name", type=str, default="libero_goal_no_noops/1.0.0",
                       help="LIBERO task name")
    parser.add_argument("--episode_idx", type=int, default=0,
                       help="Episode index to analyze (single episode)")
    parser.add_argument("--episode_range", type=str, default=None,
                       help="Episode range to analyze (e.g., '0-5' for episodes 0-5, or '0,2,4' for specific episodes)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--save_codes", type=str, default=None,
                       help="Path to save extracted codes (optional)")
    parser.add_argument("--save_video", type=str, default=None,
                       help="Path to save codes visualization video (optional)")
    parser.add_argument("--save_plot", type=str, default=None,
                       help="Path to save code sequence plot (optional)")
    parser.add_argument("--fps", type=int, default=10,
                       help="FPS for output video")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Determine episode indices to process
    if args.episode_range:
        if '-' in args.episode_range:
            # Range format: "0-5"
            start, end = map(int, args.episode_range.split('-'))
            episode_indices = list(range(start, end + 1))
        elif ',' in args.episode_range:
            # List format: "0,2,4"
            episode_indices = [int(x.strip()) for x in args.episode_range.split(',')]
        else:
            # Single number
            episode_indices = [int(args.episode_range)]
    else:
        # Single episode
        episode_indices = [args.episode_idx]
    
    print(f"Processing episodes: {episode_indices}")
    
    # Load model
    print("Loading latent action model...")
    model = load_latent_action_model(args.checkpoint, args.device)
    
    # Process each episode
    for episode_idx in episode_indices:
        print(f"\n{'='*60}")
        print(f"PROCESSING EPISODE {episode_idx}")
        print(f"{'='*60}")
        
        # Load trajectory
        print(f"Loading LIBERO trajectory: {args.task_name}, episode {episode_idx}")
        episode = load_libero_trajectory(args.data_root, args.task_name, episode_idx)
        
        if episode is None:
            print(f"❌ Failed to load episode {episode_idx}")
            continue
        
        # Process images
        print("Processing trajectory images...")
        video_sequence = process_libero_images(episode)
        
        # Extract real codes
        print("Extracting real latent action codes...")
        outputs = extract_real_codes_from_trajectory(model, video_sequence, args.device)
        
        # Analyze code quality
        analyze_code_quality(outputs)
        
        # Save codes if requested
        if args.save_codes:
            save_path = args.save_codes.replace('.npz', f'_episode{episode_idx}.npz')
            save_codes_to_file(outputs, save_path)
        
        # Create codes visualization video if requested
        if args.save_video:
            print("Creating codes visualization video...")
            video_path = f"{args.save_video}_episode{episode_idx}.mp4"
            create_codes_video(video_sequence, outputs, video_path, args.fps)
        
        # Create code sequence plot if requested
        if args.save_plot:
            print("Creating code sequence plot...")
            plot_path = f"{args.save_plot}_episode{episode_idx}.png"
            create_code_sequence_plot(outputs, plot_path)
    
    print(f"\n{'='*50}")
    print("✓ Real code analysis completed successfully!")
    print(f"Processed {len(episode_indices)} episodes: {episode_indices}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
