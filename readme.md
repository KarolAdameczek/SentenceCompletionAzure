# EY - Text Generation

**Zespół:**

1. Karol Adameczek 299231
2. Dawid Rawski 299287
3. Robert Martyka 299268



**Zadanie:**

Zbudowanie aplikacji uzupełniającej zdania w języku polskim rozpoczęte przez użytkownika - generator tekstu.



**Zbiór danych:**

[Korpus dyskursu parlamentarnego](https://kdp.nlp.ipipan.waw.pl/)



**Stos technologiczny:**

* Python3 
  * Flask
* javascript
* HTML/CSS
* Designer w Azure Machine Learning



## Funkcjonalność

1. Użytkownik ładuje stworzoną stronę internetową w przeglądarce,
2. W pole tekstowe wprowadza początek określonego zdania,
   - strona internetowa za pomocą skryptu javascript wysyła zapytanie o dokończenie aktualnej treści zdania,
   - na podstawie przesłanej treści wytrenowany model generuje drugą część zdania,
   - wygenerowana treść jest odsyłana jako odpowiedź na poprzednie zapytanie,
   - model doucza się na podstawie początku zdania podanego przez użytkownika.



## **Architektura projektu:**

![](diagram.png)

* Azure Machine Learning z Designerem, wykorzystanie bloków Web Input oraz Web Output
* App Services - Web App do hostowania strony internetowej
* Azure Storage Accouts - dane do machine learningu
* Azure Functions - preprocesowanie danych do uczenia

## Harmonogram

| Lp.  | Data       | Opis                                                         |
| ---- | ---------- | ------------------------------------------------------------ |
| 1.   | 03.12.2020 | Stworzenie opisu projektu, zebranie wymagań                  |
| 2.   | 10.12.2020 | Rozkompresowane i wczytane dane z korpusu do Azure Blob Storage, funkcja preprocesująca dane (check-point) |
| 3.   | 17.12.2020 | Wykonana analiza zbioru danych, pierwsza iteracja tworzenia aplikacji uczącej model |
| 4.   | 07.01.2021 | Druga iteracja aplikacji                                     |
| 5.   | 14.01.2021 | Aplikacja webowa ładująca poprawnie interfejs użytkownika (check-point) |
| 6.   | 21.01.2021 | Aplikacja webowa wysyłająca zapytania do modelu              |
| 7.   | 28.01.2021 | Prezentacja projektu                                         |

