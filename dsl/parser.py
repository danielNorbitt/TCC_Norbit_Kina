import sys
import json
from string import Template

class ListDSL:
    def __init__(self, json):
        self.consistency_model = json["consistency_model"]
        self.cross_sharding = bool(json["cross_sharding"])
        self.key_generator = json["key_generator"]
        
    def __str__(self):
        return "consistency_model: {0}\ncross_sharding: {1}\nkey_generator: {2}".format(self.consistency_model, self.cross_sharding, self.key_generator)

    def compile(self):
       with open('list_template.dn', 'r') as file:
            src = Template(file.read())
            result = src.safe_substitute(teste="Teste")
            self.write_file("List.dn", result)
            
    def write_file(self, file_name, content):
        with open(file_name, 'w') as file:
            file.write(content)

def get_json(json_path):
    with open(json_path, "r") as file:
        data = json.loads(file.read())
        return data

def main(file_config):
    json_content = get_json(file_config)
    list_dsl = ListDSL(json_content)
    list_dsl.compile()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_config = sys.argv[1]
    else:
        file_config = "teste.json"
    main(file_config=file_config)