from construct import Int8un, Int16un, Array, Struct
from fda import FDA
from fds import FDS


field_headers = {}
field_headers[b'@CAPTURE_INFO_02'] = Struct(
    'x' / Int16un,
    'zeros' / Array(52, Int16un),
    'aquisition year' / Int16un,
    'aquisition month' / Int16un,
    'aquisition day' / Int16un,
    'aquisition hour' / Int16un,
    'aquisition minute' / Int16un,
    'aquisition second' / Int16un
)

field_headers[b'@PATIENT_INFO_02'] = Struct(
    'patient id' / Array(32, Int8un),
    'patient given name' / Array(32, Int8un),
    'patient surname' / Array(32, Int8un),
    'zeros' / Array(8, Int8un),
    'birth data valid' / Int8un,
    'birth year' / Int16un,
    'birth month' / Int16un,
    'birth day' / Int16un,
    'zeros2' / Array(504, Int8un)
)

def read_field(oct_object, key):
    with open(oct_object.filepath, 'rb') as f:
        chunk_location, chunk_size = oct_object.chunk_dict[key]
        f.seek(chunk_location)
        raw = f.read(chunk_size)
        header = field_headers[key].parse(raw)
    return header
