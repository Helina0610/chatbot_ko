#import pandas as pd

## CSV 파일 읽기
#df = pd.read_csv("text_classification/output/round_robin_interleaved_v7.csv")

## 라벨 매핑 정의
#label_mapping = {
#    "고발" :"A411080",
#    "모범" :"A411090",
#    "변상" :"A411010",
#    "시정(금액)" :"A411030",
#    "시정(기타)" :"A411037",
#    "인사자료통보" :"A411072",
#    "주의" : "A411040",
#    "징계문책" :"A411042",
#    "통보권고" :"A411070"
#}

## 라벨값 변경
#df["label"] = df["label"].map(label_mapping)

## 결과 확인
#print(df.head())

## (선택) 변경된 CSV 저장
#df.to_csv("text_classification/output_1/datasets_v7.csv", index=False)


#import json


#filter = [{
#	"dispCd" : [
#		"A411080"
#	],
#	"test" : "asdg"
#}]


#def filtering_metadata(metadata_list: list[dict]) -> list[dict]:
#    """
#    metadata 리스트에서 각 항목의 'dispCd' 값을 'dispCd_코드': True 형식으로 변환하고,
#    원래의 'dispCd' 키는 삭제함.

#    :param metadata_list: 메타데이터 딕셔너리들의 리스트
#    :return: 변환된 메타데이터 리스트
#    """
#    for metadata in metadata_list:
#        dispCd_list = metadata.get("dispCd", [])
#        for dispCd in dispCd_list:
#            metadata[f"dispCd_{dispCd}"] = True
#        metadata.pop("dispCd", None)
    
#    print(f"metadata_list: {metadata_list}")
#    return metadata_list

#new_metadata = filtering_metadata(filter)
#print(f"new_metadata: {new_metadata}")