import jnius_config

jnius_config.add_classpath(
    "libs/jvntextpro.jar",
    "libs/vietsentiwordnet_v1.0.jar",
)

from jnius import autoclass

# Load class
OpinionFinder = autoclass("vnu.uet.vietsentiwordnet.apis.OpinionFinder")
sl = OpinionFinder.getInstance().loadModels()
ResultObject = autoclass("vnu.uet.vietsentiwordnet.objects.ResultObject")
res = ResultObject()
javaclass_String = autoclass("java.lang.String")
while True:
    text = input()
    res = sl.doSenLevel(javaclass_String(text))
    print(str(res.getResultString()))
