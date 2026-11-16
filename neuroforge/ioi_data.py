import random

NAMES = ["John", "Mary", "Tom", "Anna", "James", "Emma", "Paul", "Sara",
         "Mark", "Lisa", "David", "Kate", "Peter", "Laura", "Steve", "Julia"]
TEMPLATES = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "Then, {A} and {B} went to the park. {B} gave a ball to",
    "After {A} and {B} arrived at school, {B} handed a book to",
]

def make_pairs(model, n=50, seed=0):
    """Return a list of (clean, corrupted, correct_id, wrong_id).

    clean:     "A and B ... B gave ... to"  -> answer A
    corrupted: "A and B ... C gave ... to"  -> the repeat is gone, so the answer is ambiguous
    """
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        a, b, c = rng.sample(NAMES, 3)
        t = rng.choice(TEMPLATES)
        clean = t.format(A=a, B=b)
        # Replace only the second mention of B (the subject) with a new name C
        first, second = t.split("{B}", 1)[0], t.split("{B}", 1)[1]
        corrupted = (first + "{B}" + second.replace("{B}", "{C}", 1)).format(A=a, B=b, C=c)
        try:
            correct = model.to_single_token(" " + a)
            wrong = model.to_single_token(" " + b)
        except Exception:
            continue
        if len(model.to_tokens(clean)[0]) != len(model.to_tokens(corrupted)[0]):
            continue
        pairs.append((clean, corrupted, correct, wrong))
    return pairs