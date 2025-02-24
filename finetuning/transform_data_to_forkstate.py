import re
from datasets import load_dataset

# 1. Load the original dataset
ds = load_dataset("tsesterh/logo_data_instruct")

# 2. Define a regex pattern that matches lines like:
#    embed(""" some multiline code """, locals())
# We'll capture:
#   - The leading indentation (if any)
#   - The code inside the triple quotes
pattern = re.compile(
    r'^(?P<indent>\s*)embed\("""(?P<code>.*?)""",\s*locals\(\)\)',
    re.MULTILINE | re.DOTALL
)

def embed_to_fork_state(match):
    """
    Regex callback that transforms:
        <indent>embed(\"\"\" code... \"\"\", locals())
    into:
        <indent>with fork_state():
            <indent>    code...
    """
    indent = match.group("indent")
    code_block = match.group("code")

    # Split the embedded code into lines
    lines = code_block.split("\n")

    # Re-indent each line one level deeper than the original embed call
    # so it appears under "with fork_state():"
    # We'll prepend: indent + 4 spaces
    reindented_lines = [(indent + "    " + line if line.strip() else line)
                        for line in lines]

    # Join the re-indented lines with newlines
    transformed_code = "\n".join(reindented_lines)

    # Return a `with fork_state():` block
    return f"{indent}with fork_state():\n{transformed_code}"

def transform_completion(example):
    """
    This function is mapped over the dataset; it applies our regex to
    the 'completion' field.
    """
    old_code = example["completion"]
    new_code = pattern.sub(embed_to_fork_state, old_code)
    return {"completion": new_code}

# 3. Apply the transformation to the dataset
#    This will modify the "completion" by removing embed(...) calls
#    and replacing them with with fork_state(): blocks
ds_fork = ds.map(transform_completion)

# 4. Push the transformed dataset to Hugging Face under a new name
ds_fork.push_to_hub("tsesterh/logo_data_instruct_fork")