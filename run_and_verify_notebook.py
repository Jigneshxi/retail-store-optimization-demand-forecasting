import json
import sys
import traceback
import io
import contextlib

def execute_notebook(nb_path):
    print(f"Executing notebook: {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    import matplotlib
    matplotlib.use('Agg')
    global_env = {'display': lambda x: print(x)}
    code_count = 1
    total_code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"Total code cells to execute: {total_code_cells}")

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            raw_code = ''.join(cell.get('source', []))
            # Clean ipython magic lines like %matplotlib inline or display
            exec_lines = []
            for line in raw_code.split('\n'):
                if line.strip().startswith('%'):
                    continue
                exec_lines.append(line)
            clean_code = '\n'.join(exec_lines)

            print(f"Executing code cell {code_count}/{total_code_cells} (Cell Index {idx})...", end=' ')
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                # Add display mock if needed
                if 'display' not in global_env:
                    global_env['display'] = lambda x: print(x)

                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    exec(clean_code, global_env)

                out_text = stdout_capture.getvalue()
                err_text = stderr_capture.getvalue()
                outputs = []
                if out_text:
                    outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [line + '\n' for line in out_text.splitlines()]
                    })
                if err_text:
                    # Treat warnings as stderr output
                    outputs.append({
                        "output_type": "stream",
                        "name": "stderr",
                        "text": [line + '\n' for line in err_text.splitlines()]
                    })

                cell['execution_count'] = code_count
                cell['outputs'] = outputs
                print("SUCCESS")
            except Exception as e:
                print(f"FAILED with error: {e}")
                traceback.print_exc()
                cell['execution_count'] = code_count
                cell['outputs'] = [{
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": traceback.format_exc().splitlines()
                }]
                # Stop on error
                raise e

            code_count += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"\nAll {total_code_cells} code cells executed successfully without any error!")

if __name__ == '__main__':
    execute_notebook('Sample ML Submission Template.ipynb')
    execute_notebook('Integrated_Retail_Analytics_AML_Capstone.ipynb')
