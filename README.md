# CP-Model

<h1 align="center">
  <img src="https://github.com/swh9534/swh9534/raw/main/blueno_circle_logo.png" alt="Blueno" width="300">
  <br>
</h1>

<h4 align="center">Main language and library are as follows:</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"><br/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white">
  <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>
</p>

## 프로젝트 소개

'인터넷 게시글의 위험 카테고리 및 위험도 분석' 언어 모델 개발을 통해 사이버 범죄 예방/방지

* 게시물, 텍스트를 입력하면 해당 Input의 위험 카테고리와 위험도를 분석
```python
# model.py

Input : 웹캠으로 찍은 몰카보고싶다
                                      ********RISK REPORT*******
                                      **위험 카테고리: normal  **
                                      **위험도 : 0.54%	      **
                                      **각 카테고리별 확률:    **
                                      ** normal: 0.46         **
                                      **  마약: 0.01          **
                                      **  성범죄: 0.21        **
                                      **  해킹: 0.31          **
                                      **************************
```
* 위의 모델을 적용한 판별기 Prototype
<p align="center">
  <img src="https://github.com/user-attachments/assets/80761659-0e76-447b-8503-217fc5378f1a">
</p>

* 판별기 분석 결과에 따른 챗봇 서비스
<p align="center">
  <img src="https://github.com/user-attachments/assets/7dc85a38-9d84-4b88-81c3-6e8d807adede">
</p>

## 개발 상세 내용


---

> GitHub [@swh9534](https://github.com/swh9534) &nbsp;&middot;&nbsp;
> Velog [@Blueno](https://velog.io/@blueno/posts)
