import zlib
import json
import base64

ZIPJSON_KEY = 'base64(zip(o))'

# From https://medium.com/@busybus/zipjson-3ed15f8ea85d
# Example:
# original = {'a': "A", 'b': "B"}
# with open('test.zip', 'w') as f:
#       json.dump(json_zip(original), f, indent=4)
# with open('test.zip', 'r') as f:
#       unzipped = json_unzip(json.load(f)))


def json_unzip(zipped_json, insist=True):
    try:
        assert (zipped_json[ZIPJSON_KEY])
        assert (set(zipped_json.keys()) == {ZIPJSON_KEY})
    except:
        if insist:
            raise RuntimeError(
                "JSON not in the expected format {" + str(ZIPJSON_KEY) + ": zipstring}")
        else:
            return zipped_json

    try:
        unzipped_json = zlib.decompress(
            base64.b64decode(zipped_json[ZIPJSON_KEY]))
    except:
        raise RuntimeError("Could not decode/unzip the contents")

    try:
        loaded_json = json.loads(unzipped_json)
    except:
        raise RuntimeError("Could interpret the unzipped contents")

    return loaded_json


def json_zip(original_json):
    zipped_json = {
        ZIPJSON_KEY: base64.b64encode(
            zlib.compress(
                json.dumps(original_json).encode('utf-8')
            )
        ).decode('ascii')
    }
    return zipped_json
