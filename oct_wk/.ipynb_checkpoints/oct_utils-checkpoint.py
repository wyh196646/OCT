from construct import Int16un, Array, Struct


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


def read_field(oct_object, key):
    with open(oct_object.filepath, 'rb') as f:
        chunk_location, chunk_size = oct_object.chunk_dict[key]
        f.seek(chunk_location)
        raw = f.read(chunk_size)
        header = field_headers[key].parse(raw)
    return header
