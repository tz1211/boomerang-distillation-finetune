"""
Code adapted from https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py.

Answer checker API that uses sympy to simplify expressions and check for equality.

Utility functions for task evaluation.

Functions:
- batch_generate: Generate responses for a batch of prompts using vLLM.
- grade_answer: Grade an answer against a ground truth.
- multiple_choice_num_to_letter: Convert a number to a letter for multiple choice questions.
"""

import sympy
import signal
import regex as re
from typing import List, Optional
from pylatexenc import latex2text
from contextlib import contextmanager
from sympy.parsing import sympy_parser

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from evaluate.utils import math_normalize
from evaluate.utils.instructions_registry import INSTRUCTION_DICT
from evaluate.utils.instructions_util import IFEvalInputItem

import warnings

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations using SIGALRM."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    except TimeoutError:
        raise
    finally:
        signal.alarm(0)  # Cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    
    # Suppress pylatexenc warnings about unconfigured macros (frac is handled by math_normalize.py)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*macro.*failed its substitution.*")
        try:
            with timeout(2):  # 2 second timeout for LaTeX parsing
                expr = latex2text.LatexNodes2Text().latex_to_text(expr)
        except TimeoutError:
            # If LaTeX parsing hangs, print warning and return original expression
            print(f"WARNING: LaTeX parsing timed out after 2 seconds for expression: {expr[:100]}...")
            return expr.strip()

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            try:
                with timeout(5):  # 5 second timeout
                    simplified = sympy.simplify(sympy_diff)
                    if simplified == 0:
                        are_equal = True
            except TimeoutError:
                # If sympy.simplify() hangs, print the expressions and return False
                print(f"WARNING: sympy.simplify() timed out after 5 seconds")
                print(f"  Ground truth: {ground_truth_normalized}")
                print(f"  Given answer: {given_normalized}")
                are_equal = False
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def extra_extraction(response: str) -> str:
    """
    Extracts the final answer from a model response.
    - Finds the last occurrence of 'final answer' (case-insensitive)
    - Removes any trailing ':' or 'is'
    - Removes any trailing punctuation or whitespace
    - Removes any trailing <eos>, <end_of_turn>, <eom>, <end_of_message>,
      <eot>, <end_of_sequence>, and anything following them.
    """
    if not response:
        return ""

    lower_resp = response.lower()
    idx = lower_resp.rfind("final answer")
    if idx == -1:
        return ""

    # Extract substring after 'final answer'
    substring = response[idx + len("final answer"):]

    # Remove optional ':' or 'is'
    substring = re.sub(r"^\s*[:\-]?\s*(is\s*)?", "", substring, flags=re.IGNORECASE)

    # Remove any trailing special tokens and everything after them
    substring = re.sub(
        r"\s*<\s*(end_of_turn|eom|end_of_message|eot|end_of_sequence|eos)\s*>.*$",
        "",
        substring,
        flags=re.IGNORECASE,
    )

    # Trim trailing punctuation and whitespace
    cleaned = substring.strip()
    cleaned = re.sub(r"[.?!\s]+$", "", cleaned)

    return cleaned.strip()


def extract_last_boxed(text: str, added_answer_extraction=False) -> str:
    # Find the last occurrence of \boxed{
    start_index = text.rfind(r"\boxed{")
    if start_index == -1: # no boxed found, attempt added answer extraction
        if added_answer_extraction:
            return extra_extraction(text)
        return text
    # Move index to the character after '{'
    start_index += len(r"\boxed{")

    # Parse manually to handle nested braces
    brace_count = 1
    content = []
    for char in text[start_index:]:
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
        content.append(char)
        if brace_count == 0:
            # Remove the final '}' from the content
            content.pop()
            break
    return "".join(content).strip()


def grade_answer(given_answer: str, ground_truth: str, extract=True, added_answer_extraction=False) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False
    if extract:
        given_answer = extract_last_boxed(given_answer, added_answer_extraction=added_answer_extraction)
        ground_truth = extract_last_boxed(ground_truth)

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_instruction_following(
    input_item: IFEvalInputItem,
    response: str,
):
    """Tests response to see if instructions are followed."""
    instruction_list = input_item.instruction_id_list
    is_following_dict = {}

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in input_item.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=input_item.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_dict[instruction_id] = True
        else:
            is_following_dict[instruction_id] = False

    return is_following_dict


def multiple_choice_num_to_letter(num: int) -> str:
    """Convert a number to a letter for multiple choice questions."""
    mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
    }
    return mapping[num]


def batch_generate(
    model: LLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    apply_chat_template: bool = True,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """
    Generate responses for a batch of prompts using vLLM.
    
    Args:
        model: The vLLM model
        tokenizer: The tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        apply_chat_template: Whether to apply chat template
        system_prompt: Optional system prompt to prepend to each user message
        
    Returns:
        List of generated text responses
    """
    # Optionally format prompts using chat template
    has_chat = (
        apply_chat_template
        and getattr(tokenizer, "chat_template", None)
        and hasattr(tokenizer, "apply_chat_template")
    )
    if apply_chat_template and not has_chat:
        raise ValueError("apply_chat_template=True but tokenizer is missing a valid chat_template or apply_chat_template method.")

    if has_chat:
        formatted_prompts = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
    else:
        # If no chat template, prepend system prompt directly if provided
        if system_prompt:
            formatted_prompts = [f"{system_prompt}\n\n{prompt}" for prompt in prompts]
        else:
            formatted_prompts = prompts
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_new_tokens,
    )
    
    # Generate
    outputs = model.generate(formatted_prompts, sampling_params)
    
    # Extract responses
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text.strip())
    
    return formatted_prompts, responses