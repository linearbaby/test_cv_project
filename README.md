### Текущая версия вебсервиса: https://cv2.sufood.ru/

### UPD1
Доброй ночи, я сделал первую версию сервиса. Так как задача которую вы сказали решить довольно исследована, 
я решил сконцентрироваться на продукте и сделать решение на основе готовых компонентов. 
Плюс это было весело :). В скором времени могу переписать задачу на кастомный детектор и классификатор, 
однако без кастомного датасета, мне наврятли удастся решить задачу в разы лучше чем весь интернет.

### UPD2
В техзадании конечно ничего не сказано про детекцию лица, но я решил добавить так как обычно bb лица необходимо для получения классификации эмоции, так что фича.

В качестве улучшения текущего сервиса, можно
1) добавить сначала детектор лица, затем обученную (или обучить) классификатор эмоции на кропнутых бб лицах.
2) Просто обучить какую нибудь yolo8 в классификацию эмоции.
3) Добавить трекер (под вопросом так как в тех задании в целом нет задачи детекции)
4) Обернуть все в инференс ONNX.
