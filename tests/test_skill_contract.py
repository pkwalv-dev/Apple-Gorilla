"""Tests for a skill's calling contract.

The property under test: what AG advertises to the model is what `run()` actually
reads. This is the failure that motivated the module — a skill that passes its own
test (which calls `run()` directly) and then fails every real call, because the
advertised args key was a sentence rather than a key.
"""
from ag.skills import contract
from ag.skills.registry import Skill, _to_tool


CELSIUS = "def run(args, broker=None):\n    return str(args['celsius'] * 9 / 5 + 32)\n"


# --- reading the truth out of the code --------------------------------------
def test_keys_come_from_the_code_not_the_claim():
    assert contract.keys_read_by_run(CELSIUS) == ["celsius"]
    assert contract.keys_read_by_run("def run(args, b=None):\n"
                                     "    return args.get('url', '') + args['path']"
                                     ) == ["url", "path"]


def test_the_bag_parameter_need_not_be_called_args():
    assert contract.keys_read_by_run(
        "def run(payload, broker=None):\n    return payload['q']") == ["q"]


def test_unparseable_or_dynamic_code_reads_no_keys():
    assert contract.keys_read_by_run("def run(args):\n    return args[") == []
    assert contract.keys_read_by_run("def run(args, b=None):\n"
                                     "    k = 'x'\n    return args[k]") == []


def test_required_keys_exclude_those_with_a_default():
    code = ("def run(args, b=None):\n"
            "    return args['must'] + args.get('may', '')\n")
    assert contract.required_keys(code) == ["must"]
    assert contract.keys_read_by_run(code) == ["must", "may"]


# --- normalising what the authoring model claimed ---------------------------
def test_a_description_is_reduced_to_its_key():
    # The exact string that shipped a broken skill.
    assert contract.normalize_arg(
        "celsius (float): the temperature in degrees Celsius") == "celsius"
    assert contract.normalize_arg("url") == "url"
    assert contract.normalize_arg("url (str)") == "url"
    assert contract.normalize_arg('args["path"]') == "path"


def test_prose_with_no_identifier_in_it_yields_nothing():
    assert contract.normalize_arg("the temperature to convert") == ""
    assert contract.normalize_arg("") == ""


# --- reconciling the two ----------------------------------------------------
def test_the_code_overrules_a_wrong_declaration():
    primary, keys, err = contract.reconcile("temperature", CELSIUS)
    assert (primary, keys, err) == ("celsius", ["celsius"], None)


def test_a_declaration_the_code_agrees_with_leads_the_example():
    primary, keys, err = contract.reconcile(
        "path", "def run(args, b=None):\n    return args['url'] + args['path']")
    assert primary == "path" and keys == ["path", "url"] and err is None


def test_a_declaration_is_trusted_only_when_the_code_says_nothing():
    primary, keys, err = contract.reconcile("query", "def run(args, b=None):\n"
                                                     "    return str(args)")
    assert (primary, keys, err) == ("query", ["query"], None)


def test_no_contract_at_all_is_an_error_not_a_guess():
    _, _, err = contract.reconcile("the thing to do", "def run(args, b=None):\n"
                                                      "    return str(args)")
    assert err and "no usable args key" in err


# --- what the model is shown ------------------------------------------------
def test_the_advertised_example_uses_the_real_keys():
    sk = Skill(name="c2f", description="Converts C to F.", arg="celsius",
               args=["celsius"], optional=[])
    assert '"args":{"celsius":"..."}' in _to_tool(sk, lambda a, b: "").desc


def test_optional_keys_are_named_but_not_required_in_the_example():
    sk = Skill(name="fetch", description="Fetches.", arg="url",
               args=["url", "timeout"], optional=["timeout"])
    desc = _to_tool(sk, lambda a, b: "").desc
    assert '"args":{"url":"..."}' in desc        # the call the model must make
    assert "optional: timeout" in desc           # and what else it may pass


def test_a_legacy_manifest_is_repaired_on_load():
    """A skill registered before the contract was checked must not stay unusable."""
    sk = Skill.from_dict({"name": "c2f", "description": "d",
                          "arg": "celsius (float): the temperature"})
    assert sk.arg == "celsius" and sk.keys() == ["celsius"]
    assert '"celsius":"..."' in sk.example_args()
