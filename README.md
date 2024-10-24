# 강화학습 기반 V2G 전력 거래 이익 최대화

## 1. 프로젝트 소개

### 프로젝트 개요
본 프로젝트는 V2G(Vehicle-to-Grid) 환경에서 전력거래 이익을 최대화하기 위한 강화학습 기반 솔루션을 제안합니다. EV2Gym 시뮬레이션 환경을 활용하여 다양한 강화학습 알고리즘의 성능을 평가하고, 최적의 충방전 전략을 개발했습니다.

### 연구 배경

- 신재생 에너지 비중 확대에 따른 전력 공급 과잉 문제 발생
- 전기차 보급 확대에 따른 V2G 시스템 활용 가능성 증가

### 연구 목표

- V2G 환경에서 강화학습을 통한 최적 충방전 전략 개발
- EV 사용자와 그리드 관리자 모두의 이익을 고려한 통합 솔루션 제시
- 배터리 수명과 충전 만족도를 고려한 경제적 운영 방안 도출

## 2. 팀소개

<table>
  <tr>
    <td align="center">
      <a href="https://www.github.com/jasper200207">
        <img src="https://github.com/jasper200207.png" width="80" alt="main manager"/>
        <br/><b>김도균</b>
      </a>
    </td>
    <td>
      <li>PPO, SAC 알고리즘</li>
      <li>Action Sampling 기법 구상</li>
      <li>그리드 관리자 관점 시나리오 구상</li>
      <li>논문 연구 및 강화학습 구상</li>
      <li>샘플 코드 테스트 모니터링 및 시뮬레이션 환경 구축</li>
    </td>
    <td align="center">
      <a href="https://www.github.com/Leonheart0910">
        <img src="https://github.com/Leonheart0910.png" width="80" alt="main manager"/>
        <br/><b>배레온</b>
      </a>
    </td>
    <td>
      <li>EV 사용자 관점 시나리오 구상</li>
      <li>V2G 모델 수학적 분석</li>
      <li>V2G 모델 설계</li>
      <li>TRPO 알고리즘</li>
      <li>다중 목적 최적화 구상</li>
    </td>
  </tr>
</table>

## 3. 시스템 구상도

<img width="1606" alt="image" src="https://github.com/user-attachments/assets/e292cfcf-588d-4094-b573-95d63ba0032b">

## 4. 소개영상

<!-- 나중에 제출한 영상 유튜브로 바꾸기 -->
[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://youtu.be/zh_gQ_lmLqE)

## 5. 참고문헌

- [EV2Gym: A Flexible V2G Simulator for EV Smart Charging Research and Benchmarking](https://doi.org/10.48550/arXiv.2404.01849) - Orfanoudakis, S., Diaz-Londono, C., Yılmaz, Y. E., Palensky, P., & Vergara, P. P. (2024). *arXiv:2404.01849 [cs.SE]*.

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://doi.org/10.48550/arXiv.1801.01290) - Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *International Conference on Machine Learning (ICML)*.

- [Trust Region Policy Optimization](https://doi.org/10.48550/arXiv.1502.05477) - Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). *International Conference on Machine Learning (ICML)*.

- [Proximal Policy Optimization Algorithms](https://doi.org/10.48550/arXiv.1707.06347) - Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
