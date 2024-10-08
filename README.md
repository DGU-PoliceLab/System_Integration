## 소개

이 저장소는 동국대학교 대학원이 치안현장 맞춤형 연구개발사업(폴리스랩 2.0)을 위해 개발한 인공지능 모델의 통합 시스템입니다.

<details><summary>폴리스랩 2.0이란?
</summary>
  - 국민, 경찰, 연구자 등이 협업하여 치안 현장에서 발생하는 문제를 발굴하고 첨단과학기술과 ICT융합을 통해 문제해결 및 실증
  - 연구자와 사용자(경찰)간 상호작용을 촉진하기 위해 실제 환경에서 기술개발이 가능한 실증 실험실(폴리스랩*) 구축
  - * 치안을 뜻하는 폴리스(Police)와 리빙랩(Living-Lab)의 합성어
  - 치안 현장의 문제해결을 위해 실제 적용 및 검증할 수 있도록 현장에서 실증연구를 강화하여 완결성 제고
  ![PoliceLab 2.0](https://www.nrf.re.kr/file/image?path=S5u0o7mp43XMnSXx5OUq4zSOZuFLG/hVD2gLAtrKTJ0=&name=%EF%BC%8FeqnXmMLaSGZZF%EF%BC%8FQMhbBmI/tBCI9Q0SGVwTKMjiV7wM=)   
</details>

### 주요 기능

#### 실시간 낙상 감지
[🔗 CSDC 연구실](https://sites.google.com/dgu.ac.kr/csdc/)

유치장에서 발생하는 넘어짐, 쓰러짐 등의 낙상 사고를 실시간으로 감지합니다. 이를 통해 신속한 대응이 가능하며, 수감자의 안전을 보다 효과적으로 보호할 수 있습니다.

#### 실시간 자살자해 감지
[🔗 PLASS 연구실](https://sites.google.com/dgu.ac.kr/plass/home)

유치장에서 발생하는 자살 및 자해 행동을 실시간으로 감지합니다. 이를 통해 빠른 대응이 가능하며, 수감자의 생명과 안전을 더욱 효과적으로 보호할 수 있습니다.

#### 실시간 감정 분석
[🔗 HRI 연구실](http://hri.dongguk.edu/)

유치장에서 수감 중인 수감자의 감정을 실시간으로 3단계로 분석합니다. 이를 통해 수감자의 심리 상태를 파악하고, 필요한 지원이나 개입을 적시에 제공할 수 있습니다.

## 설치 및 실행

### 설치

본 시스템은 도커 환경을 권장합니다.
[🐳 Dockerfile](https://github.com/DGU-PoliceLab/System_Integration/blob/main/Dockerfile)

1. 도커 이미지 생성
   ```bash
   docker build -t {TAG_NAME}:{TAG_VERSION} .
   ```
2. 도커 컨테이너 생성
   ```bash
   docker run --gpus all -it --name {CONTAINER_NAME} {TAG_NAME}:{TAG_VERSION}
   ```
3. 도커 진입 후, 설정 진행
   ```bash
   bash /System_Integration/_Scripts/setting.sh
   ```

## 실행
```bash
python run.py
```
