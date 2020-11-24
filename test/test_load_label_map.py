
from utils.utils import load_label_list

label_map = dict()

_label_map = load_label_list()
if type(label_map) is not dict:
    _label_map = {}
    raise TypeError('label_list is not correct')
    sys.exit(0)
else:
    print(_label_map)