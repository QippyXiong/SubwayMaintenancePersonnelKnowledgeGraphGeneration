from __init__ import DataBaseManager

def clearClasses(manager: DataBaseManager, class_name: str = "类型"):
    query = f'''
        MATCH (n:{ class_name })
        DETACH DELETE n
    '''
    manager.execute_query(query)

def importOutlineClassFile(manager: DataBaseManager, file_path: str, class_name: str = "类型"):
    r""" 用于导入大纲文件 """
    class_names = []
    sub_class_relations = []
    current_layer_level = 0
    parent_stack = []
    previous_class_name = ''
    with open(file_path, "r", encoding="UTF-8") as fd:
        for line in fd:
            previous_layer_level = current_layer_level
            # get current level
            current_layer_level = 0

            for char in line:
                if char == '\t':
                    current_layer_level += 1
                else: break
            # append name to list
            class_names.append(line[current_layer_level:-1])

            if current_layer_level == 0: # 没有子类关系
                previous_class_name = line[:-1]
                continue
            if current_layer_level > previous_layer_level:
                parent_stack.append(previous_class_name)
            elif current_layer_level < previous_layer_level:
                parent_stack.pop()
        
            previous_class_name = line[current_layer_level:-1]            
            sub_class_relations.append((parent_stack[-1:][0], previous_class_name))
    
    print(class_names, sub_class_relations)
    for class_label in class_names:
        query = f'''
            CREATE (:{class_name} {{ label: "{ class_label }" }})
        '''
        r = manager.execute_query(query)
        print("?//", r)

    for sub_class_relation in sub_class_relations:
        query = f'''
            MATCH (a:{ class_name }), (b:{ class_name })
            WHERE a.label = "{ sub_class_relation[0] }" AND b.label = "{ sub_class_relation[1] }"
            CREATE (a)-[:是父类型]->(b)
            CREATE (b)-[:是子类型]->(a)
        '''
        manager.execute_query(query)


if __name__ == '__main__':
    import os
    current_dir = os.path.dirname(__file__)
    print(current_dir)
    project_dir = ''.join([ item + os.path.sep for item in current_dir.split(os.path.sep)[:-4]]) # ../../..
    print(project_dir)
    outline_file_path = os.path.join(project_dir, "MaintenanceSimulation", "kg", "人员能力知识图谱_实体结点分类.txt")
    manager = DataBaseManager()
    importOutlineClassFile(manager, outline_file_path)
    manager.close()
    