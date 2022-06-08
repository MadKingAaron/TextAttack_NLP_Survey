import os
import json

json_file = open("integrated_recipes_classifcation.json", 'r')

info = json.load(json_file)
print(info)

for recipe in info['recipes']:
    for model in info['models'].keys():
        csv_name = "%s_%s.csv" %(recipe, model)
        if not os.path.exists(csv_name):
            if info['models'][model] == 'imdb' and recipe != 'clare':
                command = 'textattack attack --model %s --attack-recipe %s --log-to-csv %s --dataset-from-huggingface %s' %(model, recipe, csv_name, info['models'][model])
                os.system(command)
        

#os.system("textattack -h")