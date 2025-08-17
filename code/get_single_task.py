from task_dict import tasks_dic

i = 0
tasks = []
for task in tasks_dic["muv"]:
    tasks.append(tasks_dic["muv"][i])
    i += 1

f = open("muv", "w")

for index, task in enumerate(tasks):
    f.write(f"'muv{index+1}' : ['{task}'],\n")

f.close()

i = 0
tasks = []
for task in tasks_dic["tox21"]:
    tasks.append(tasks_dic["tox21"][i])
    i += 1

f = open("tox21", "w")

for index, task in enumerate(tasks):
    f.write(f"'tox21{index+1}' : ['{task}'],\n")

f.close()
