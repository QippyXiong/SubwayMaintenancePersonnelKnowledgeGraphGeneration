## Ontology使用说明

如果是windows操作系统，导入Ontology文件前，请确保你的neo4j和Ontology.ttl文件放在一个磁盘下。

首先，安装neosemantics，在桌面版Db启动的Plugins中选择进行安装
[neosemantics](https://github.com/neo4j-labs/neosemantics)

然后创建一个constraint，在neo4j的DB浏览器控制命令行中，输入以下SQL创建

```cypher
CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE
```

导入Ontology文件，输入以下SQL，修改其中的dir为你的文件夹路径
```cypher
CALL n10s.rdf.import.fetch("file:///dir/Ontology.ttl", "Turtle");
```

如果你是在windows操作系统上，仍建议你使用分隔符`/`，一个例子是
```cypher
CALL n10s.rdf.import.fetch("file:///D:/dev/SubwayMaintenancePersonnelKnowledgeGraphGeneration/kg/OWL/Ontology.ttl", "Turtle");
```

导入Ontology文件后，类名、实体名称会出现Prestege中设置的Ontology iri，需要去除的话，输入以下cypher语句以去除掉所有导入的结点前面的链接，语句中的链接更换成你想去除的链接，长度25需要需改为你实际链接的字符个数

```cypher
match(n) where n.uri=~"http://www.listalina.iri#.*" set n.uri=substring(n.uri,25) return n
```
