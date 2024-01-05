from methods.sprompt import SPrompts
from methods.prefix_prompt_tuning import PrefixPromptTuning
def get_model(model_name, args):
    name = model_name.lower()
    options = {
        'sprompts': SPrompts,
        'prefix_one_prompt':PrefixPromptTuning,
        }
    return options[name](args)

