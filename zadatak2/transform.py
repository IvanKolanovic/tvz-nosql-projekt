#!/usr/bin/env python3
import csv
import json
import os
import sys

def transform_opensignals_to_csv(input_file_path):
    """
    Transform an OpenSignals format file to CSV format.
    
    Args:
        input_file_path (str): Path to the OpenSignals format file
    
    Returns:
        str: Path to the generated CSV file
    """
    print(f"Transforming {input_file_path} to CSV format...")
    
    # Create output file path with the same name but .csv extension
    base_name = os.path.splitext(input_file_path)[0]
    output_file_path = f"{base_name}.csv"
    
    metadata = None
    column_names = None
    data_rows = []
    
    # Read the OpenSignals file
    with open(input_file_path, 'r') as file:
        # Process header section
        for line in file:
            line = line.strip()
            
            # Skip OpenSignals header line
            if line.startswith("# OpenSignals"):
                continue
                
            # Extract metadata
            elif line.startswith("# {"):
                try:
                    # Remove the "# " prefix and parse JSON
                    json_str = line[2:]
                    metadata = json.loads(json_str)
                    
                    # Extract column names from metadata
                    device_id = list(metadata.keys())[0]
                    column_names = metadata[device_id]["column"]
                    print(f"Found {len(column_names)} columns: {', '.join(column_names)}")
                except Exception as e:
                    print(f"Error parsing metadata: {e}")
                    
            # End of header section
            elif line.startswith("# EndOfHeader"):
                print("Header processing complete, starting data transformation...")
                break
                
            # Skip other comment lines
            elif line.startswith("#"):
                continue
        
        # Process data lines
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            # Split tab-separated values
            values = line.split("\t")
            
            # Only add rows with the correct number of columns
            if column_names and len(values) == len(column_names):
                data_rows.append(values)
            elif column_names:
                print(f"Warning: Skipping row with {len(values)} values (expected {len(column_names)})")
    
    # Write to CSV file
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        if column_names:
            writer.writerow(column_names)
            
        # Write data rows
        writer.writerows(data_rows)
    
    print(f"Transformation complete! CSV file created at: {output_file_path}")
    print(f"Processed {len(data_rows)} data rows")
    
    return output_file_path

def main():
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        input_file_path = sys.argv[1]
    else:
        # Default to data.txt in the current directory
        input_file_path = "zadatak2/data.txt"
    
    # Check if the file exists
    if not os.path.isfile(input_file_path):
        print(f"Error: File '{input_file_path}' not found.")
        sys.exit(1)
    
    # Transform the file
    output_file_path = transform_opensignals_to_csv(input_file_path)
    print(f"Successfully created {output_file_path}")

if __name__ == "__main__":
    main()
