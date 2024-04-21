

import re
import xml.etree.ElementTree as ET
import json

def extract_function_calls_from_xml(completion):
    """Extracts function calls from XML formatted string."""
    completion = completion.strip()
    pattern = r"((.*?))"
    match = re.search(pattern, completion, re.DOTALL)
    if not match:
        print("No match found")  
        return None

    xml_content = match.group(1)
    if xml_content:  
        root = ET.fromstring(xml_content)
        functions = root.findall("functioncall")
        return [json.loads(fn.text) for fn in functions]
    else:
        print("No XML content to parse")  
        return None
