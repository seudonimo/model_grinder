import onnx
import json
import argparse
from onnx import shape_inference
from onnx2json import convert as o2j_convert
from json2onnx import convert as j2o_convert

# onnx 모델을 입력 받아 json으로 뽑아낸 다음, 모든 레이어에 대해 unit model을 생성해준다.
'''
1. Set the shape by static inference
2. Check the in/out for each unit graph
3. apply them to template
4. save
'''

def loadModel(file_path):
    return onnx.load(file_path)

def setModelAsStatic(model, modelName, output='./'):
    # Shape inference를 실행하여 모델을 업데이트합니다.
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, '{}{}Static.onnx'.format(output, modelName))
    return inferred_model

def getModelAsONNX(jsonModel):
    return j2o_convert(json_dict=jsonModel)

def getModelAsJSON(model):
    return o2j_convert(onnx_graph=model,)

def getDefaultModel(model):
    new_dict = dict()
    for k in model.keys():
        new_dict[k] = model[k]
    return new_dict

def grinderModel(model):
    jsonModel = getModelAsJSON(model)
    
    for i in range(len(jsonModel['graph']['node'])):
        unit_item = jsonModel['graph']['node'][i]
        unit_model = getDefaultModel(jsonModel)
        if i == 0:
            unit_model['graph']['input'] = jsonModel['graph']['input']
            unit_model['graph']['output'] = jsonModel['graph']['valueInfo'][i]
        elif i == len(jsonModel['graph']['node']) -1:
            unit_model['graph']['input'] = jsonModel['graph']['valueInfo'][i]
            unit_model['graph']['output'] = jsonModel['graph']['output']
        else:
            unit_model['graph']['input'] = jsonModel['graph']['valueInfo'][i]
            unit_model['graph']['input'] = jsonModel['graph']['valueInfo'][i+1]
        unit_model['graph']['name'] = 'unit_{}'.format(i)
        unit_model['graph']['node'] = [unit_item]
        #unit_model['graph']['initializer'] = []
        #unit_model['graph']['valueInfo'] = []
        onnxModel = getModelAsONNX(unit_model)
        onnx.save(onnxModel, 'unit_model_{}.onnx'.format(i))

    return

def main(file_path):
    
    model = loadModel(file_path)
    modelName = file_path.split('/')[-1].split('.')[-2]
    staticModel = setModelAsStatic(model, modelName)
    grinderModel(staticModel)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_file',
        type=str,
        required=True,
        help='Input ONNX model path. (*.onnx)'
    )
    args = parser.parse_args()
    main(args.input_file)
    