from typing import Union

# Apply gemini-style chat template for instruct models
def apply_chat_template(user_text: str, model_text: Union[str, None] = None,
                        apply_bos: bool = False, apply_eos: bool = False):
    templated = "<bos>" if apply_bos else ""
    templated = templated + f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

    if model_text is not None:
        templated = templated + model_text

    if apply_eos:
        templated = templated + "<eos>"

    return templated