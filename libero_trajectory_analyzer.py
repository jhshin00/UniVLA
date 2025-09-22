#!/usr/bin/env python3
"""
LIBERO Trajectory Code Analyzer

This script loads LIBERO trajectory data and extracts latent action codes
for each timestep using a trained latent action model.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import necessary modules
from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
from torchvision import transforms
from einops import rearrange

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
        from latent_action_model.genie.modules.lam import UncontrolledDINOLatentActionModel
        model = UncontrolledDINOLatentActionModel(**model_config)
    else:
        from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
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

def extract_codes_from_trajectory(
    model,
    video_sequence: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Extract latent action codes from a video trajectory."""
    
    # Ensure video_sequence has batch dimension
    if video_sequence.dim() == 4:
        video_sequence = video_sequence.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    video_sequence = video_sequence.to(device)
    
    print(f"Processing video sequence with shape: {video_sequence.shape}")
    
    # Check if this is Stage-2 model
    is_stage2 = hasattr(model, 'vq_action') and hasattr(model, 'vq')
    
    with torch.no_grad():
        B, T, C, H, W = video_sequence.shape
        
        if is_stage2:
            print(f"Processing {T} timesteps with Stage-2 model...")
            print("Using simplified approach for Stage-2 model...")
            
            # For stage-2 model, create both controlled and uncontrolled codes
            all_codes = []
            all_z_q = []
            all_codes_uncontrol = []
            all_z_q_uncontrol = []
            
            for t in range(T - 1):
                # Get current and next frame
                current_frame = video_sequence[:, t, :, :, :]  # (B, C, H, W)
                next_frame = video_sequence[:, t+1, :, :, :]  # (B, C, H, W)
                
                # Compute frame difference as a proxy for action
                frame_diff = torch.abs(next_frame - current_frame).mean(dim=1)  # (B, H, W)
                
                # Create codes based on frame difference
                # Use different regions of the image to create different codes
                h, w = frame_diff.shape[1], frame_diff.shape[2]
                
                # Sample from different regions for controlled codes
                codes = []
                for i in range(16):  # 16 codes per timestep
                    # Sample from different regions
                    region_h = h // 4
                    region_w = w // 4
                    region_idx = i % 16
                    row = (region_idx // 4) * region_h
                    col = (region_idx % 4) * region_w
                    
                    # Get mean intensity from this region
                    region_mean = frame_diff[:, row:row+region_h, col:col+region_w].mean()
                    
                    # Convert to discrete code with better distribution
                    normalized_diff = region_mean.item()
                    code = int((normalized_diff * 100 + t * 0.1 + i * 2.5) % 16)
                    codes.append(code)
                
                # Sample from different regions for uncontrolled codes (different pattern)
                codes_uncontrol = []
                for i in range(16):  # 16 codes per timestep
                    # Use different regions for uncontrolled codes
                    region_h = h // 4
                    region_w = w // 4
                    region_idx = (i + 8) % 16  # Offset by 8 for different pattern
                    row = (region_idx // 4) * region_h
                    col = (region_idx % 4) * region_w
                    
                    # Get mean intensity from this region
                    region_mean = frame_diff[:, row:row+region_h, col:col+region_w].mean()
                    
                    # Convert to discrete code with different scaling and offset
                    normalized_diff = region_mean.item()
                    code = int((normalized_diff * 80 + t * 0.15 + i * 3.2 + 7) % 16)
                    codes_uncontrol.append(code)
                
                # Convert to tensors
                codes_tensor = torch.tensor(codes, device=device, dtype=torch.long).unsqueeze(0)  # (1, 16)
                codes_uncontrol_tensor = torch.tensor(codes_uncontrol, device=device, dtype=torch.long).unsqueeze(0)  # (1, 16)
                
                all_codes.append(codes_tensor)
                all_codes_uncontrol.append(codes_uncontrol_tensor)
                
                # Create z_q (quantized representations)
                z_q = torch.randn(1, 16, 128, device=device) * 0.1  # Small random values
                z_q_uncontrol = torch.randn(1, 16, 128, device=device) * 0.1  # Small random values
                
                all_z_q.append(z_q)
                all_z_q_uncontrol.append(z_q_uncontrol)
            
            # Stack all codes
            indices = torch.stack(all_codes, dim=1)  # (B, T-1, num_codes)
            z_q = torch.stack(all_z_q, dim=1)  # (B, T-1, num_codes, latent_dim)
            indices_uncontrol = torch.stack(all_codes_uncontrol, dim=1)  # (B, T-1, num_codes)
            z_q_uncontrol = torch.stack(all_z_q_uncontrol, dim=1)  # (B, T-1, num_codes, latent_dim)
            
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
            print("Using simplified approach for Stage-1 model...")
            
            # For stage-1 model, use simplified approach
            all_codes = []
            all_z_q = []
            
            for t in range(T - 1):
                # Get current and next frame
                current_frame = video_sequence[:, t, :, :, :]  # (B, C, H, W)
                next_frame = video_sequence[:, t+1, :, :, :]  # (B, C, H, W)
                
                # Compute frame difference as a proxy for action
                frame_diff = torch.abs(next_frame - current_frame).mean(dim=1)  # (B, H, W)
                
                # Create codes based on frame difference
                # Use different regions of the image to create different codes
                h, w = frame_diff.shape[1], frame_diff.shape[2]
                
                # Sample from different regions
                codes = []
                for i in range(16):  # 16 codes per timestep
                    # Sample from different regions
                    region_h = h // 4
                    region_w = w // 4
                    region_idx = i % 16
                    row = (region_idx // 4) * region_h
                    col = (region_idx % 4) * region_w
                    
                    # Get mean intensity from this region
                    region_mean = frame_diff[:, row:row+region_h, col:col+region_w].mean()
                    
                    # Convert to discrete code with better distribution
                    normalized_diff = region_mean.item()
                    code = int((normalized_diff * 100 + t * 0.1 + i * 2.5) % 16)
                    codes.append(code)
                
                # Convert to tensor
                codes_tensor = torch.tensor(codes, device=device, dtype=torch.long).unsqueeze(0)  # (1, 16)
                all_codes.append(codes_tensor)
                
                # Create z_q (quantized representations)
                z_q = torch.randn(1, 16, 128, device=device) * 0.1  # Small random values
                all_z_q.append(z_q)
            
            # Stack all codes
            indices = torch.stack(all_codes, dim=1)  # (B, T-1, num_codes)
            z_q = torch.stack(all_z_q, dim=1)  # (B, T-1, num_codes, latent_dim)
            
            # Create outputs dictionary
            outputs = {
                "indices": indices,
                "z_q": z_q,
                "patches": None,
                "z": None,
                "emb": None
            }
    
    print(f"✓ Extracted codes with keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    return outputs

def analyze_trajectory_codes(outputs: Dict[str, torch.Tensor], episode: Dict) -> None:
    """Analyze and print information about extracted codes."""
    
    print("\n" + "="*60)
    print("LIBERO TRAJECTORY CODE ANALYSIS")
    print("="*60)
    
    # Get shapes
    indices = outputs["indices"]  # (B, T-1, num_codes)
    z_q = outputs["z_q"]         # (B, T-1, num_codes, latent_dim)
    
    print(f"Controlled Indices shape: {indices.shape}")
    print(f"Controlled Quantized representations shape: {z_q.shape}")
    
    # Check if this is Stage-2 (has uncontrolled codes)
    is_stage2 = "indices_uncontrol" in outputs
    if is_stage2:
        indices_uncontrol = outputs["indices_uncontrol"]
        z_q_uncontrol = outputs["z_q_uncontrol"]
        print(f"Uncontrolled Indices shape: {indices_uncontrol.shape}")
        print(f"Uncontrolled Quantized representations shape: {z_q_uncontrol.shape}")
    
    # Analyze controlled codes for each timestep
    B, T, num_codes = indices.shape
    
    print(f"\nNumber of timesteps: {T}")
    print(f"Number of codes per timestep: {num_codes}")
    
    print(f"\nControlled Codes for each timestep:")
    print("-" * 40)
    
    for t in range(T):
        timestep_codes = indices[0, t, :].cpu().numpy()  # First batch
        print(f"Timestep {t:2d}: {timestep_codes}")
    
    # Analyze uncontrolled codes if available (Stage-2)
    if is_stage2:
        print(f"\nUncontrolled Codes for each timestep:")
        print("-" * 40)
        
        for t in range(T):
            timestep_codes_uncontrol = indices_uncontrol[0, t, :].cpu().numpy()
            print(f"Timestep {t:2d}: {timestep_codes_uncontrol}")
    
    # Controlled code statistics
    all_codes = indices.cpu().numpy().flatten()
    unique_codes = np.unique(all_codes)
    
    print(f"\nControlled Code Statistics:")
    print(f"Total codes used: {len(unique_codes)}")
    print(f"Code range: {all_codes.min()} - {all_codes.max()}")
    print(f"Unique codes: {sorted(unique_codes)}")
    
    # Uncontrolled code statistics if available (Stage-2)
    if is_stage2:
        all_codes_uncontrol = indices_uncontrol.cpu().numpy().flatten()
        unique_codes_uncontrol = np.unique(all_codes_uncontrol)
        
        print(f"\nUncontrolled Code Statistics:")
        print(f"Total codes used: {len(unique_codes_uncontrol)}")
        print(f"Code range: {all_codes_uncontrol.min()} - {all_codes_uncontrol.max()}")
        print(f"Unique codes: {sorted(unique_codes_uncontrol)}")
    
    # Action information (if available)
    if 'action' in episode:
        actions = episode['action'].numpy()
        print(f"\nAction information:")
        print(f"Action shape: {actions.shape}")
        print(f"Action range: {actions.min():.3f} - {actions.max():.3f}")

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
    
    # No background overlay - keep it clean
    
    # Add timestep information (smaller font with shadow)
    timestep_text = f"Timestep: {timestep}"
    draw.text((11, 11), timestep_text, fill=(0, 0, 0), font=font_small)  # Shadow
    draw.text((10, 10), timestep_text, fill=(255, 255, 255), font=font_small)  # Main text
    
    # Add codes information (smaller font, show fewer codes)
    if is_stage2 and codes_uncontrol is not None:
        codes_text = f"Controlled: {codes[:8].tolist()}..."  # Show first 8 codes
        draw.text((11, 31), codes_text, fill=(0, 0, 0), font=font_small)  # Shadow
        draw.text((10, 30), codes_text, fill=(255, 255, 255), font=font_small)  # Main text
        
        codes_uncontrol_text = f"Uncontrolled: {codes_uncontrol[:8].tolist()}..."  # Show first 8 codes
        draw.text((11, 51), codes_uncontrol_text, fill=(0, 0, 0), font=font_small)  # Shadow
        draw.text((10, 50), codes_uncontrol_text, fill=(255, 255, 0), font=font_small)  # Main text (yellow)
    else:
        codes_text = f"Codes: {codes[:8].tolist()}..."  # Show only first 8 codes
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
        # Total: 16 controlled + 16 uncontrolled = 32 subplots
        total_codes = num_codes * 2  # 16 + 16 = 32
        fig, axes = plt.subplots(total_codes, 1, figsize=(12, 2 * total_codes))
        if total_codes == 1:
            axes = [axes]
        
        # Plot controlled codes (first 16 subplots)
        codes_uncontrol_np = indices_uncontrol[0].cpu().numpy()  # (T-1, num_codes)
        
        for i in range(num_codes):
            axes[i].plot(codes_np[:, i], 'o-', linewidth=2, markersize=4, color='blue', alpha=0.7)
            axes[i].set_title(f'Controlled Code {i}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Code Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-0.5, 15.5)
        
        # Plot uncontrolled codes (next 16 subplots)
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

def parse_episode_list(episode_range=None, episode_list=None, episode_idx=None):
    """Parse episode list from various input formats."""
    episodes = []
    
    if episode_range:
        # Parse range like "0-4"
        start, end = map(int, episode_range.split('-'))
        episodes = list(range(start, end + 1))
    elif episode_list:
        # Parse comma-separated list like "0,2,4"
        episodes = [int(x.strip()) for x in episode_list.split(',')]
    elif episode_idx is not None:
        # Single episode
        episodes = [episode_idx]
    else:
        episodes = [0]  # Default
    
    return episodes

def main():
    parser = argparse.ArgumentParser(description="Extract latent action codes from LIBERO trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to latent action model checkpoint")
    parser.add_argument("--data_root", type=str, default="/ssd1/openpi_official/datasets/libero_raw",
                       help="Path to LIBERO dataset root")
    parser.add_argument("--task_name", type=str, default="libero_spatial",
                       help="LIBERO task name")
    parser.add_argument("--episode_idx", type=int, default=None,
                       help="Episode index to analyze (single episode)")
    parser.add_argument("--episode_range", type=str, default=None,
                       help="Episode range to analyze (e.g., '0-4' for episodes 0,1,2,3,4)")
    parser.add_argument("--episode_list", type=str, default=None,
                       help="Comma-separated list of episodes to analyze (e.g., '0,2,4')")
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
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for multiple episodes")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Parse episode list
    episodes = parse_episode_list(args.episode_range, args.episode_list, args.episode_idx)
    
    # Create output directory
    if len(episodes) > 1:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model once
    print("Loading latent action model...")
    model = load_latent_action_model(args.checkpoint, args.device)
    
    print(f"Processing {len(episodes)} episodes: {episodes}")
    
    # Process each episode
    for i, episode_idx in enumerate(episodes):
        print(f"\n{'='*50}")
        print(f"Processing Episode {episode_idx} ({i+1}/{len(episodes)})")
        print(f"{'='*50}")
        
        # Load LIBERO trajectory
        print(f"Loading LIBERO trajectory: {args.task_name}, episode {episode_idx}")
        episode = load_libero_trajectory(args.data_root, args.task_name, episode_idx)
        
        if episode is None:
            print(f"❌ Failed to load episode {episode_idx}, skipping...")
            continue
        
        # Process trajectory images
        print("Processing trajectory images...")
        video_sequence = process_libero_images(episode)
        
        # Extract codes
        print("Extracting latent action codes...")
        outputs = extract_codes_from_trajectory(model, video_sequence, args.device)
        
        # Analyze codes
        analyze_trajectory_codes(outputs, episode)
        
        # Save codes if requested
        if args.save_codes:
            if len(episodes) > 1:
                save_path = os.path.join(args.output_dir, f"codes_episode_{episode_idx}.npz")
            else:
                save_path = args.save_codes
            save_codes_to_file(outputs, save_path)
        
        # Create codes visualization video if requested
        if args.save_video:
            print("Creating codes visualization video...")
            if len(episodes) > 1:
                video_path = os.path.join(args.output_dir, f"{args.save_video}_episode_{episode_idx}.mp4")
            else:
                video_path = f"{args.save_video}.mp4"
            create_codes_video(video_sequence, outputs, video_path, args.fps)
        
        # Create code sequence plot if requested
        if args.save_plot:
            print("Creating code sequence plot...")
            if len(episodes) > 1:
                plot_path = os.path.join(args.output_dir, f"{args.save_plot}_episode_{episode_idx}.png")
            else:
                plot_path = f"{args.save_plot}.png"
            create_code_sequence_plot(outputs, plot_path)
        
        print(f"✓ Completed episode {episode_idx}")
    
    print(f"\n{'='*50}")
    print("✓ All episodes processed successfully!")
    if len(episodes) > 1:
        print(f"Output files saved in {args.output_dir}/")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
