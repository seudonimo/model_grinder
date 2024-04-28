import os
import onnx
import json
import argparse
from onnx import shape_inference
from onnx2json import convert as o2j_convert
from json2onnx import convert as j2o_convert
import copy

# onnx 모델을 입력 받아 json으로 뽑아낸 다음, 모든 레이어에 대해 unit model을 생성해준다.
'''
1. Set the shape by static inference
2. Check the in/out for each unit graph
3. apply them to template
4. save
'''



def loadModel(file_path: str) -> onnx.onnx_ml_pb2.ModelProto:
    return onnx.load(file_path)

def setModelAsStatic(model: onnx.onnx_ml_pb2.ModelProto, 
                     modelName: str,
                     output: str='./'
                     ) -> onnx.onnx_ml_pb2.ModelProto:
    # Shape inference를 실행하여 모델을 업데이트합니다.
    inferred_model = shape_inference.infer_shapes(model)
    #onnx.save(inferred_model, '{}{}_Static.onnx'.format(output, modelName))
    return inferred_model

def getModelAsONNX(jsonModel: dict) -> onnx.onnx_ml_pb2.ModelProto:
    return j2o_convert(json_dict=jsonModel)

def getModelAsJSON(model: onnx.onnx_ml_pb2.ModelProto) -> dict:
    return o2j_convert(onnx_graph=model,)

def getDefaultModel(model: onnx.onnx_ml_pb2.ModelProto) -> dict:
    new_dict = dict()
    for k in model.keys():
        new_dict[k] = copy.deepcopy(model[k])
    return new_dict

def grinderModel(model: onnx.onnx_ml_pb2.ModelProto) -> list:
    
    # Get the model as JSON dictionary
    jsonModel = getModelAsJSON(model)
    '''
    노드 리스트에서 특정 노드 차례일때...
    그 노드의 in/out이 graph의 in/out으로 복사해주고,
    혹시나 initializer(TensorProto)에 해당 노드의 weight/bias가 있는지 확인해준다.(이건 input의 이름으로 검색한다)
    혹시나2 valueInfo에서도 in/out의 이름을 검색해서 있으면 생성/챙겨준다.
    
    1. 모델마다 graph의 input/output 설정해주고
    2. 노드 리스트에서 그 노드의 
    '''
    modelLength = len(jsonModel['graph']['node'])
    modelList = list()
    for i in range(modelLength):
        if 'docString' in jsonModel['graph']['node'][i].keys():
            jsonModel['graph']['node'][i]['docString'] = ''
        unit_item = jsonModel['graph']['node'][i]
        unit_model = getDefaultModel(jsonModel)
        
        
        # Set input and output for unit model
        if i == 0:
            unit_model['graph']['input'] = jsonModel['graph']['input']
            if 'valueInfo' in jsonModel['graph'].keys():
                unit_model['graph']['output'] = [jsonModel['graph']['valueInfo'][i]]
        elif i == len(jsonModel['graph']['node']) -1:
            if 'valueInfo' in jsonModel['graph'].keys():
                unit_model['graph']['input'] = [jsonModel['graph']['valueInfo'][i-1]]
            unit_model['graph']['output'] = jsonModel['graph']['output']
        else:
            if 'valueInfo' in jsonModel['graph'].keys():
                unit_model['graph']['input'] = [jsonModel['graph']['valueInfo'][i-1]]
                unit_model['graph']['output'] = [jsonModel['graph']['valueInfo'][i]]
        unit_model['graph']['name'] = 'unit_{}'.format(i)
        unit_model['graph']['node'] = [unit_item]

        # Weight, and Bias
        for input_name in unit_item['input']:
            for item in jsonModel['graph']['initializer']:
                if input_name == item['name']:
                    unit_model['graph']['initializer'].append(item)

        unit_model['graph']['valueInfo'] = []
        modelList.append(unit_model)
    return modelList

def saveModel(modelList, folderName):

    for idx, unitModel in enumerate(modelList):
        unitName = '{}_{}_{}'.format(
            idx,
            folderName.split('/')[-1],
            unitModel['graph']['node'][0]['opType']
        )
        with open('{}/{}.json'.format(folderName, unitName), 'w') as f:
            json.dump(unitModel, f)
        onnxModel = getModelAsONNX(unitModel)
        onnx.save(onnxModel, '{}/{}.onnx'.format(folderName, unitName))
    return

def main(file_path: str, args) -> bool:
    
    model = loadModel(file_path)
    modelName = file_path.split('/')[-1].split('.')[-2]
    
    # Generate Folder
    modelName = file_path.split('/')[-1].split('.')[-2]
    folderName = '{}/{}'.format(args.output_folder, modelName)
    os.makedirs(folderName, exist_ok=True)

    staticModel = setModelAsStatic(model, modelName)
    modelList = grinderModel(staticModel)

    saveModel(modelList, folderName)
    
    return True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_file',
        type=str,
        default='./sample/mnist.onnx',
        help='Input ONNX model path. (*.onnx)'
    )
    parser.add_argument(
        '-of',
        '--output_folder',
        type=str,
        default='./RESULT',
        help=''
    )
    args = parser.parse_args()
    main(args.input_file, args)
    
