import json

notebook_path = "/home/ubuntu/CIFO_EXTENDED_Project/notebooks/02_multi_processor_analysis_executed.ipynb"
full_stdout = []
error_output = []
found_output = False

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = json.load(f)

    last_executed_code_cell_outputs = None
    for cell in reversed(notebook_content.get('cells', [])):
        if cell.get('cell_type') == 'code' and cell.get('execution_count') is not None:
            if cell.get('outputs'):
                # Check if this cell contains the call to main()
                source_code = "".join(cell.get('source', []))
                if "if __name__ == \"__main__\":" in source_code and "main()" in source_code:
                    last_executed_code_cell_outputs = cell.get('outputs')
                    break
    
    if last_executed_code_cell_outputs:
        found_output = True
        for output_item in last_executed_code_cell_outputs:
            if output_item.get('output_type') == 'stream' and output_item.get('name') == 'stdout':
                full_stdout.extend(output_item.get('text', []))
            elif output_item.get('output_type') == 'error':
                error_output.append("--- ERROR OUTPUT ---")
                error_output.extend(output_item.get('traceback', []))
    else:
        # Fallback: try to get outputs from the very last cell with execution_count if specific main cell not found
        for cell in reversed(notebook_content.get('cells', [])):
            if cell.get('cell_type') == 'code' and cell.get('execution_count') is not None:
                if cell.get('outputs'):
                    last_executed_code_cell_outputs = cell.get('outputs')
                    break
        if last_executed_code_cell_outputs:
            found_output = True
            for output_item in last_executed_code_cell_outputs:
                if output_item.get('output_type') == 'stream' and output_item.get('name') == 'stdout':
                    full_stdout.extend(output_item.get('text', []))
                elif output_item.get('output_type') == 'error':
                    error_output.append("--- ERROR OUTPUT ---")
                    error_output.extend(output_item.get('traceback', []))

except Exception as e:
    full_stdout.append(f"Error processing notebook: {str(e)}\n")

if full_stdout:
    print("--- Full Standard Output ---")
    for line in full_stdout:
        print(line, end='')
    found_output = True # Ensure this is true if stdout was populated

if error_output:
    print("\n--- Full Error Output ---")
    for line in error_output:
        print(line) # Traceback lines already have newlines
    found_output = True # Ensure this is true if stderr was populated

if not found_output:
    print("No stdout or stderr found for the main execution cell or error in processing.")

