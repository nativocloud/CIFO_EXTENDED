import json

notebook_path = "/home/ubuntu/CIFO_EXTENDED_Project/notebooks/02_multi_processor_analysis_executed.ipynb"
full_stdout = []

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = json.load(f)
    
    last_executed_cell_output = None
    # Iterate through cells to find the one that likely ran main()
    # This is often the last code cell with an execution_count and outputs.
    for cell in reversed(notebook_content.get('cells', [])):
        if cell.get('cell_type') == 'code':
            ec = cell.get('execution_count')
            # Check if this cell was executed and has outputs
            if ec is not None and cell.get('outputs'):
                # A simple heuristic: if it contains the typical start message of main()
                for output_item in cell.get('outputs', []):
                    if output_item.get('output_type') == 'stream' and output_item.get('name') == 'stdout':
                        text_output = "".join(output_item.get('text', []))
                        if "Multiprocessing Script for" in text_output and "runs execution started" in text_output:
                            last_executed_cell_output = cell['outputs']
                            break
                if last_executed_cell_output:
                    break
    
    if last_executed_cell_output:
        for output_item in last_executed_cell_output:
            if output_item.get('output_type') == 'stream' and output_item.get('name') == 'stdout':
                full_stdout.extend(output_item.get('text', []))
            elif output_item.get('output_type') == 'error':
                full_stdout.append("--- ERROR OUTPUT ---\n")
                full_stdout.extend(output_item.get('traceback', []))

except Exception as e:
    print(f"Error processing notebook: {e}")

if full_stdout:
    print("--- Full Standard Output of Main Execution ---")
    for line in full_stdout:
        print(line, end='')
else:
    print("No stdout found for the main execution cell or error in processing.")

