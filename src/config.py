from omegaconf import OmegaConf

default_config = dict(
    # Secret key for signing tokens (replace with your own secret key)
    DETECTION_MODEL="${oc.env:DETECTION_MODEL,yunet}",
)


######################################### VALIDATION ########################################
######################################### VALIDATION ########################################
######################################### VALIDATION ########################################
def validate_configs(default_config_schema, config, mismatched={}, prefix=""):
    """
    Recursively validate keys and types for two dictionary configs.
    """
    # Get keys for each config
    keys1 = set(default_config_schema.keys())
    keys2 = set(config.keys())

    # Check for keys present in one config but not the other
    extra_keys_in_config1 = keys1 - keys2
    extra_keys_in_config2 = keys2 - keys1

    if extra_keys_in_config1 or extra_keys_in_config2:
        for mism in extra_keys_in_config1:
            mismatched[prefix + mism] = "not declared"
        for mism in extra_keys_in_config2:
            mismatched[prefix + mism] = "extra var provided"

    # Check the types and values recursively
    for key in keys1.intersection(keys2):
        value1 = default_config_schema[key]
        value2 = config[key]

        # If both values are dictionaries, recursively validate them
        if isinstance(value1, dict) and isinstance(value2, dict):
            validate_configs(value1, value2, mismatched, prefix + key + ".")
        else:
            # Check if types match
            if type(value2) != type(value1):
                mismatched[prefix + key] = f"types mismatch: {type(value2)} != {type(value1)}"

    return mismatched


default_config = OmegaConf.create(default_config)
# possibly to comment if no cli is used
# config = OmegaConf.merge(default_config, OmegaConf.from_cli())
config = OmegaConf.merge(default_config, OmegaConf.create({}))
OmegaConf.resolve(config)
OmegaConf.resolve(default_config)

# Validate merged config against the schema
mismatched = validate_configs(OmegaConf.to_container(default_config), OmegaConf.to_container(config))
if len(mismatched) > 0:
    raise ValueError(f"Configs have different keys: {mismatched}")