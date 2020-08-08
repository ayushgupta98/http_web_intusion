def count_alpha_numeric_spchars(string):
    """return the number of digits, characters and special characters present in a string"""
    string = str(string)
    list = [0,0,0] # digit, char, special characters
    string_check= re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    for ch in string:
        if ch.isdigit():
            list[0] += 1
        elif (string_check.search(ch) != None):
            list[2] += 1
        else:
            list[1] += 1
    return list[0], list[1], list[2]


def length_path(url):
    """returns the length of path minus the host (http://localhost:80/)"""
    return "/".join(url.split('/')[3::])

def label_fix(string):
    """encoding the feature label"""
    l = string.split(',')
    if 'anom' in l:
        return 0
    else:
        return 1

def content_length(string):
    """returns the length of feature content_length"""
    if string == '':
        return 0
    else:
        return int(string)

def content_type(string):
    """returns the length of feature countent_type"""
    if string == '':
        return 0
    else:
        return len(str(string))

def number_of_keywords(string):
    """return the number of keyword present in the argument"""
    # list of selected keywords
    keywords = ['SELECT','DELETE','DROP','UPDATE','INSERT','FROM','WHERE','LIKE','TABLE','JOIN','ORDER','waitfor', '||','|','&&','%3B','/bin','/usr','/passwd','echo','/shadow',';','$','grep','curl','wget',\
                'which', 'script','alert','img','mouse','onerror','style','svg','onload','body','iframe','xml']
    new_str = str(string).upper()
    count= 0
    for key in keywords:
        if key in new_str:
            count += 1
    #if count > 0:
    #    return 1
    #else:
    #    return 0
    return count

def encode_method(string):
    """Encoding the feature method"""
    if string == 'GET':
        return 1, 0, 0
    elif string == 'POST':
        return 0, 1, 0
    else:
        return 0, 0, 1

def label_encode(string):
    """Encoding the feature label"""
    if string == 'anom':
        return 1
    else:
        return 0