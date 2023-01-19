# generate pipeline

matilda platform pipeline에서 사용이 가능한 파이프라인을 생성합니다.   
   
`component-files` : 컴포넌트룰 구성하는 `.yaml`파일이 구성되어 있습니다.   
`python-files` : 컴포넌트를 생성하기 위한 코드로 구성되어 있습니다.   
   
필요한 컴포넌트는 `python-files`내에서 생성하여 사용합니다.   
필요한 컴포넌트를 모두 생성한 후 `create-test-pipeline.py`를 통하여 컴포넌트를 연결합니다.   

```
python run.py
```
`run.py`를 실행하면 `python-files`내에 있는 모든 python 코드를 수행하여 각 컴포넌트에 해당하는 yaml파일을 생성합니다.
실행 결과로 `test_pipeline.yaml`이 생성됩니다.