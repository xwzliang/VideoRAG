import json
import re
import os
import argparse
from typing import Dict, List, Tuple, Optional

def extract_timestamps(text: str) -> List[Tuple[str, float, float]]:
    """Extract timestamps from text in various formats.
    Returns list of (original_text, start_time, end_time) tuples."""
    # Match patterns like [0.00s -> 3.64s], (0.00s -> 3.64s), [0.00 - 3.64], (0.00 - 3.64)
    patterns = [
        r'\[(\d+\.?\d*)s?\s*->\s*(\d+\.?\d*)s?\]',  # [0.00s -> 3.64s]
        r'\((\d+\.?\d*)s?\s*->\s*(\d+\.?\d*)s?\)',  # (0.00s -> 3.64s)
        r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]',       # [0.00 - 3.64]
        r'\((\d+\.?\d*)\s*-\s*(\d+\.?\d*)\)'        # (0.00 - 3.64)
    ]
    
    timestamps = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            timestamps.append((match.group(0), start_time, end_time))
    
    return timestamps

def adjust_timestamps(text: str, offset: float) -> str:
    """Adjust all timestamps in text by adding offset and unify format to [start -> end]."""
    timestamps = extract_timestamps(text)
    
    # Sort by position in text to avoid replacing shorter matches first
    timestamps.sort(key=lambda x: text.find(x[0]), reverse=True)
    
    for original, start, end in timestamps:
        # Always use the format [start -> end]
        new_timestamp = f"[{start + offset:.2f}s -> {end + offset:.2f}s]"
        text = text.replace(original, new_timestamp)
    
    return text

def process_video_segments(json_path: str, target_video: Optional[str] = None) -> str:
    """Process video segments from JSON file and combine their contents with adjusted timestamps.
    
    Args:
        json_path: Path to the JSON file containing video segments
        target_video: Optional video name to process only that video's segments
    
    Returns:
        Combined content with adjusted timestamps
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    combined_content = []
    
    # Filter videos if target_video is specified
    videos_to_process = {target_video: data[target_video]} if target_video else data
    
    # Process each video
    for video_name, segments in videos_to_process.items():
        if target_video and video_name != target_video:
            continue
            
        # Sort segments by index to ensure correct order
        sorted_segments = sorted(segments.items(), key=lambda x: int(x[0]))
        
        for segment_idx, segment_data in sorted_segments:
            # Get segment start time from the time field (format: "start-end")
            start_time = float(segment_data['time'].split('-')[0])
            
            # Adjust timestamps in content
            adjusted_content = adjust_timestamps(segment_data['content'], start_time)
            
            # Add video name and segment info as header
            header = f"\n=== Video: {video_name}, Segment {segment_idx} (Time: {segment_data['time']}) ===\n"
            combined_content.append(header + adjusted_content)
    
    return "\n".join(combined_content)

def main():
    parser = argparse.ArgumentParser(description='Process video segments and adjust timestamps.')
    parser.add_argument('--working_dir', type=str, default=os.path.expanduser("~/videos/videorag-workdir"),
                      help='Working directory containing the video segments JSON file')
    parser.add_argument('--video_name', type=str, default=None,
                      help='Optional: Process only segments from this video')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Optional: Custom output file name (default: combined_video_content.txt)')
    
    args = parser.parse_args()
    
    # Construct paths
    json_path = os.path.join(args.working_dir, "kv_store_video_segments.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Video segments JSON file not found at {json_path}")
    
    # Set default output filename if not provided
    if args.output_file is None:
        if args.video_name:
            args.output_file = f"combined_video_content_{args.video_name}.txt"
        else:
            args.output_file = "combined_video_content.txt"
    
    output_path = os.path.join(args.working_dir, args.output_file)
    
    # Process the segments
    combined_content = process_video_segments(json_path, args.video_name)
    
    # Save the combined content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    print(f"Processed content saved to {output_path}")

if __name__ == "__main__":
    main()
