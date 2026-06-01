import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '02_rl_training.ipynb')

def main():
    print(f"Modifying notebook: {NOTEBOOK_PATH}")
    if not os.path.exists(NOTEBOOK_PATH):
        print("Notebook not found!")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                if 'step_penalty=-0.001' in line:
                    line = line.replace('step_penalty=-0.001', 'step_penalty=-0.01')
                    modified = True
                    print("Found and replaced step_penalty=-0.001 with step_penalty=-0.01")
                new_source.append(line)
            cell['source'] = new_source

    if modified:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print("Notebook saved successfully.")
    else:
        print("No match found in notebook cells.")

if __name__ == "__main__":
    main()
