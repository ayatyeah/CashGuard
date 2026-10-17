export type Language='ru'|'en'|'kk';
const dictionary:Record<string,[string,string]>={
"Открыть видео":["Open video","Бейнені ашу"],
  "Ошибка обработки": [
    "Processing failed",
    "Өңдеу қатесі"
  ],
  "В очереди": [
    "Queued",
    "Кезекте"
  ],
  "Анализируется": [
    "Analyzing",
    "Талдануда"
  ],
  "Требует проверки": [
    "Needs review",
    "Тексеру қажет"
  ],
  "Без нарушений": [
    "No violations",
    "Бұзушылықтар жоқ"
  ],
  "Выберите MP4, MOV или WebM размером до 500 МБ.": [
    "Choose an MP4, MOV or WebM file up to 500 MB.",
    "Көлемі 500 МБ-қа дейінгі MP4, MOV немесе WebM файлын таңдаңыз."
  ],
  "Проверено видео": [
    "Videos reviewed",
    "Тексерілген бейнелер"
  ],
  "Завершённые проверки": [
    "Completed reviews",
    "Аяқталған тексерулер"
  ],
  "Требуют проверки": [
    "Need review",
    "Тексеруді қажет етеді"
  ],
  "Возможные нарушения": [
    "Possible violations",
    "Ықтимал бұзушылықтар"
  ],
  "Признаки не обнаружены": [
    "No signs detected",
    "Белгілер анықталған жоқ"
  ],
  "В обработке": [
    "Processing",
    "Өңделуде"
  ],
  "В очереди и на анализе": [
    "Queued and analyzing",
    "Кезекте және талдауда"
  ],
  "Контроль кассовых операций": [
    "Cash operations monitoring",
    "Кассалық операцияларды бақылау"
  ],
  "РАБОЧЕЕ ПРОСТРАНСТВО": [
    "WORKSPACE",
    "ЖҰМЫС КЕҢІСТІГІ"
  ],
  "Обзор": [
    "Overview",
    "Шолу"
  ],
  "Анализ видео": [
    "Video analysis",
    "Бейнені талдау"
  ],
  "Локальный контур": [
    "Local environment",
    "Жергілікті орта"
  ],
  "Видео обрабатывается внутри вашей инфраструктуры.": [
    "Video is processed within your infrastructure.",
    "Бейне сіздің инфрақұрылымыңызда өңделеді."
  ],
  "Без облачных AI-сервисов": [
    "No cloud AI services",
    "Бұлттық AI сервистерінсіз"
  ],
  "Настройки подключения": [
    "Connection settings",
    "Қосылым параметрлері"
  ],
  "Войти в аккаунт": [
    "Sign in to your account",
    "Аккаунтқа кіру"
  ],
  "Переключить меню": [
    "Toggle navigation",
    "Мәзірді ашу немесе жабу"
  ],
  "Рабочее пространство": [
    "Workspace",
    "Жұмыс кеңістігі"
  ],
  "Сервис подключён": [
    "Service connected",
    "Сервис қосылды"
  ],
  "Нет подключения": [
    "Not connected",
    "Қосылым жоқ"
  ],
  "ВИДЕОАНАЛИТИКА · БЕЗОПАСНОСТЬ": [
    "VIDEO ANALYTICS · SECURITY",
    "БЕЙНЕ ТАЛДАУ · ҚАУІПСІЗДІК"
  ],
  "Обзор проверок": [
    "Review overview",
    "Тексерулерге шолу"
  ],
  "Всё, что важно для контроля кассовых операций.": [
    "Everything you need to monitor cash operations.",
    "Кассалық операцияларды бақылауға қажетті ақпарат."
  ],
  "Загрузите запись и проверьте наблюдаемые события.": [
    "Upload a recording and review observable events.",
    "Жазбаны жүктеп, көрінетін оқиғаларды тексеріңіз."
  ],
  "Новый анализ": [
    "New analysis",
    "Жаңа талдау"
  ],
  "Демонстрационный режим.": [
    "Demo mode.",
    "Демонстрациялық режим."
  ],
  "Результаты синтетические; загруженное видео не анализируется.": [
    "Results are synthetic; the uploaded video is not analyzed.",
    "Нәтижелер жасанды; жүктелген бейне талданбайды."
  ],
  "Личное рабочее пространство": [
    "Your personal workspace",
    "Жеке жұмыс кеңістігі"
  ],
  "Войдите или зарегистрируйтесь, чтобы хранить свои видео и результаты.": [
    "Sign in or register to store your videos and results.",
    "Бейнелер мен нәтижелерді сақтау үшін кіріңіз немесе тіркеліңіз."
  ],
  "Войти": [
    "Sign in",
    "Кіру"
  ],
  "Скрыть ошибку": [
    "Dismiss error",
    "Қатені жасыру"
  ],
  "Активность проверок": [
    "Review activity",
    "Тексеру белсенділігі"
  ],
  "Количество обработанных видео за неделю": [
    "Videos processed over the last week",
    "Соңғы аптада өңделген бейнелер саны"
  ],
  "7 дней": [
    "7 days",
    "7 күн"
  ],
  "Проверено": [
    "Reviewed",
    "Тексерілді"
  ],
  "РЕЗУЛЬТАТЫ АНАЛИЗА": [
    "ANALYSIS RESULTS",
    "ТАЛДАУ НӘТИЖЕЛЕРІ"
  ],
  "Контроль в деталях": [
    "Review breakdown",
    "Тексеру мәліметтері"
  ],
  "без нарушений": [
    "without violations",
    "бұзушылықсыз"
  ],
  "Выводы сервиса помогают проверяющему принять решение.": [
    "Results help the reviewer make an informed decision.",
    "Нәтижелер тексерушіге шешім қабылдауға көмектеседі."
  ],
  "Новая проверка": [
    "New review",
    "Жаңа тексеру"
  ],
  "Одна запись — полный разбор событий": [
    "One recording, a complete event review",
    "Бір жазба — оқиғаларды толық талдау"
  ],
  "Перетащите видео сюда": [
    "Drop your video here",
    "Бейнені осында сүйреп әкеліңіз"
  ],
  "МБ · готово к загрузке": [
    "MB · ready to upload",
    "МБ · жүктеуге дайын"
  ],
  "или выберите файл на вашем устройстве": [
    "or choose a file on your device",
    "немесе құрылғыңыздан файл таңдаңыз"
  ],
  "Выбрать другое": [
    "Choose another",
    "Басқасын таңдау"
  ],
  "Выбрать видео": [
    "Choose video",
    "Бейнені таңдау"
  ],
  "MP4, MOV, WebM · до 500 МБ": [
    "MP4, MOV, WebM · up to 500 MB",
    "MP4, MOV, WebM · 500 МБ-қа дейін"
  ],
  "Загрузка видео…": [
    "Uploading video…",
    "Бейне жүктелуде…"
  ],
  "Начать анализ": [
    "Start analysis",
    "Талдауды бастау"
  ],
  "Видео остаётся в локальном контуре": [
    "Video stays in the local environment",
    "Бейне жергілікті ортада сақталады"
  ],
  "ОТ ВИДЕО К РЕШЕНИЮ": [
    "FROM VIDEO TO DECISION",
    "БЕЙНЕДЕН ШЕШІМГЕ ДЕЙІН"
  ],
  "Как проходит проверка": [
    "How reviews work",
    "Тексеру қалай өтеді"
  ],
  "Загрузите запись": [
    "Upload a recording",
    "Жазбаны жүктеңіз"
  ],
  "Выберите видео работы кассира.": [
    "Choose a video of the cashier at work.",
    "Кассирдің жұмыс бейнесін таңдаңыз."
  ],
  "Дождитесь анализа": [
    "Wait for the analysis",
    "Талдауды күтіңіз"
  ],
  "Статус обновляется автоматически.": [
    "The status updates automatically.",
    "Күйі автоматты түрде жаңартылады."
  ],
  "Проверьте результат": [
    "Review the result",
    "Нәтижені тексеріңіз"
  ],
  "Изучите события, таймкоды и доказательства.": [
    "Examine events, timecodes and evidence.",
    "Оқиғаларды, уақыт белгілерін және дәлелдерді қараңыз."
  ],
  "Результат — подсказка для проверки человеком, а не окончательное заключение.": [
    "Results support human review and are not a final conclusion.",
    "Нәтиже — адамның тексеруіне көмек, түпкілікті қорытынды емес."
  ],
  "РЕЗУЛЬТАТ ПРОВЕРКИ": [
    "REVIEW RESULT",
    "ТЕКСЕРУ НӘТИЖЕСІ"
  ],
  "Закрыть результат": [
    "Close result",
    "Нәтижені жабу"
  ],
  "ДЕМО": [
    "DEMO",
    "ДЕМО"
  ],
  "Скачать JSON": [
    "Download JSON",
    "JSON жүктеп алу"
  ],
  "Видео ожидает своей очереди.": [
    "The video is waiting in the queue.",
    "Бейне өз кезегін күтуде."
  ],
  "Идёт анализ видео. Это может занять несколько минут.": [
    "Analyzing video. This may take a few minutes.",
    "Бейне талдануда. Бұл бірнеше минутқа созылуы мүмкін."
  ],
  "Нарушение не обнаружено": [
    "No violation detected",
    "Бұзушылық анықталған жоқ"
  ],
  "В доступной записи не выявлено признаков возможных нарушений.": [
    "No signs of possible violations were found in the recording.",
    "Жазбада ықтимал бұзушылық белгілері анықталған жоқ."
  ],
  "Синтетическая иллюстрация": [
    "Synthetic illustration",
    "Жасанды иллюстрация"
  ],
  "Кадр-доказательство": [
    "Evidence frame",
    "Дәлел кадры"
  ],
  "Демо-иллюстрация": [
    "Demo illustration",
    "Демо иллюстрация"
  ],
  "Открыть источник ↗": [
    "Open source ↗",
    "Дереккөзді ашу ↗"
  ],
  "Нормативный источник не предоставлен анализатором.": [
    "The analyzer did not provide a regulatory source.",
    "Талдаушы нормативтік дереккөзді ұсынбады."
  ],
  "Последние проверки": [
    "Recent reviews",
    "Соңғы тексерулер"
  ],
  "История анализов": [
    "Analysis history",
    "Талдау тарихы"
  ],
  "Записи и результаты в одном месте": [
    "Recordings and results in one place",
    "Жазбалар мен нәтижелер бір жерде"
  ],
  "Все проверки": [
    "All reviews",
    "Барлық тексерулер"
  ],
  "Все видео": [
    "All videos",
    "Барлық бейнелер"
  ],
  "Найти видео…": [
    "Find a video…",
    "Бейнені іздеу…"
  ],
  "Поиск видео": [
    "Search videos",
    "Бейнелерді іздеу"
  ],
  "ВИДЕО": [
    "VIDEO",
    "БЕЙНЕ"
  ],
  "ДАТА ЗАГРУЗКИ": [
    "UPLOADED",
    "ЖҮКТЕЛГЕН КҮНІ"
  ],
  "СТАТУС": [
    "STATUS",
    "КҮЙІ"
  ],
  "СОБЫТИЯ": [
    "EVENTS",
    "ОҚИҒАЛАР"
  ],
  "Демонстрационная проверка": [
    "Demo review",
    "Демонстрациялық тексеру"
  ],
  "Открыть": [
    "Open",
    "Ашу"
  ],
  "Видео не найдены": [
    "No videos found",
    "Бейнелер табылған жоқ"
  ],
  "Здесь появятся ваши проверки": [
    "Your reviews will appear here",
    "Тексерулеріңіз осында көрсетіледі"
  ],
  "Попробуйте изменить поиск или фильтр.": [
    "Try changing your search or filter.",
    "Іздеу шартын немесе сүзгіні өзгертіп көріңіз."
  ],
  "Загрузите первое видео, чтобы начать контроль кассовых операций.": [
    "Upload your first video to start monitoring cash operations.",
    "Кассалық операцияларды бақылауды бастау үшін алғашқы бейнені жүктеңіз."
  ],
  "Загрузить видео": [
    "Upload video",
    "Бейнені жүктеу"
  ],
  "Показано": [
    "Showing",
    "Көрсетілгені"
  ],
  "из": [
    "of",
    "/"
  ],
  "Данные хранятся локально": [
    "Data stored locally",
    "Деректер жергілікті сақталады"
  ],
  "Интеллектуальный контроль операций": [
    "Intelligent operations monitoring",
    "Операцияларды зияткерлік бақылау"
  ],
  "Проверка на основе наблюдаемых действий": [
    "Review based on observable actions",
    "Көрінетін әрекеттерге негізделген тексеру"
  ],
  "Закрыть": [
    "Close",
    "Жабу"
  ],
  "Вход в CashGuard": [
    "Sign in to CashGuard",
    "CashGuard жүйесіне кіру"
  ],
  "Создать аккаунт": [
    "Create account",
    "Аккаунт жасау"
  ],
  "Ваши видео и результаты доступны только вашему аккаунту.": [
    "Your videos and results are only accessible to your account.",
    "Бейнелеріңіз бен нәтижелеріңіз тек сіздің аккаунтыңызға қолжетімді."
  ],
  "Войдите, чтобы сохранять видео и результаты в личном пространстве.": [
    "Sign in to save videos and results in your personal workspace.",
    "Бейнелер мен нәтижелерді жеке кеңістікте сақтау үшін кіріңіз."
  ],
  "Хранилище": [
    "Storage",
    "Деректер қоймасы"
  ],
  "Выйти": [
    "Sign out",
    "Шығу"
  ],
  "Регистрация": [
    "Register",
    "Тіркелу"
  ],
  "Пароль": [
    "Password",
    "Құпиясөз"
  ],
  "Не менее 8 символов": [
    "At least 8 characters",
    "Кемінде 8 таңба"
  ],
  "Не удалось выйти": [
    "Could not sign out. Please try again.",
    "Шығу мүмкін болмады. Қайталап көріңіз."
  ],
  "Язык интерфейса": [
    "Interface language",
    "Интерфейс тілі"
  ],
  "Тёмная тема": [
    "Dark theme",
    "Қараңғы тақырып"
  ],
  "Светлая тема": [
    "Light theme",
    "Ашық тақырып"
  ],
  "Не удалось сохранить настройки": [
    "Could not save settings. Please try again.",
    "Параметрлер сақталмады. Қайталап көріңіз."
  ],
  "CashGuard — контроль кассовых операций": [
    "CashGuard — cash operations monitoring",
    "CashGuard — кассалық операцияларды бақылау"
  ],
  "Ошибка сервера": [
    "Server error",
    "Сервер қатесі"
  ],
  "Не удалось прочитать историю": [
    "Could not load history",
    "Тарихты жүктеу мүмкін болмады"
  ],
  "Видео не найдено": [
    "Video not found",
    "Бейне табылған жоқ"
  ],
  "Ошибка чтения": [
    "Could not read the result",
    "Нәтижені оқу мүмкін болмады"
  ],
  "Некорректная загрузка или файл больше 500 МБ": [
    "Invalid upload or file larger than 500 MB",
    "Жүктеу қате немесе файл көлемі 500 МБ-тан үлкен"
  ],
  "Поле video обязательно": [
    "A video file is required",
    "Бейнефайл қажет"
  ],
  "Поддерживаются MP4, MOV и WebM": [
    "MP4, MOV and WebM are supported",
    "MP4, MOV және WebM форматтары қолданылады"
  ],
  "Размер видео должен быть от 1 байта до 500 МБ": [
    "Video size must be between 1 byte and 500 MB",
    "Бейне көлемі 1 байттан 500 МБ-қа дейін болуы керек"
  ],
  "Файл не является поддерживаемым видео": [
    "This file is not a supported video",
    "Бұл бейне форматына қолдау көрсетілмейді"
  ],
  "Ошибка чтения видео": [
    "Could not read the video",
    "Бейнені оқу мүмкін болмады"
  ],
  "Ошибка создания задания": [
    "Could not create the job",
    "Тапсырманы жасау мүмкін болмады"
  ],
  "Ошибка сохранения видео": [
    "Could not save the video",
    "Бейнені сақтау мүмкін болмады"
  ],
  "Не удалось поставить видео в очередь": [
    "Could not queue the video",
    "Бейнені кезекке қою мүмкін болмады"
  ],
  "Failed to fetch": [
    "Cannot connect to the server",
    "Серверге қосылу мүмкін болмады"
  ]
};
const errors:Record<string,[string,string,string]>={
"network_error":["Не удалось подключиться к серверу","Cannot connect to the server","Серверге қосылу мүмкін болмады"],
  "invalid_credentials": [
    "Неверный email или пароль",
    "Incorrect email or password",
    "Email немесе құпиясөз қате"
  ],
  "account_exists": [
    "Этот email уже зарегистрирован",
    "This email is already registered",
    "Бұл email тіркелген"
  ],
  "invalid_registration": [
    "Укажите корректный email и пароль от 8 до 72 байт",
    "Enter a valid email and a password of 8–72 bytes",
    "Дұрыс email және 8–72 байттан тұратын құпиясөз енгізіңіз"
  ],
  "session_expired": [
    "Сессия истекла. Войдите снова.",
    "Session expired. Please sign in again.",
    "Сессия аяқталды. Қайта кіріңіз."
  ],
  "too_many_attempts": [
    "Слишком много попыток. Повторите через минуту.",
    "Too many attempts. Try again in a minute.",
    "Тым көп әрекет. Бір минуттан кейін қайталаңыз."
  ],
  "server_error": [
    "Ошибка сервера",
    "Server error",
    "Сервер қатесі"
  ],
  "preferences_failed": [
    "Не удалось сохранить настройки",
    "Could not save settings",
    "Параметрлер сақталмады"
  ],
  "invalid_preferences": [
    "Некорректные настройки",
    "Invalid settings",
    "Параметрлер қате"
  ]
};
Object.assign(dictionary,{"API-ключ подключён":["API key connected","API кілті қосылды"],"Подключить API":["Connect API","API қосу"],"Подключить":["Connect","Қосу"],"Отключить":["Disconnect","Ажырату"],"Общее рабочее пространство":["Shared workspace","Ортақ жұмыс кеңістігі"],"Подключите API-ключ, чтобы загружать видео и просматривать результаты.":["Connect your API key to upload videos and view results.","Бейне жүктеу және нәтижелерді көру үшін API кілтін қосыңыз."],"Введите API-ключ сервера. Видео и результаты хранятся в общем рабочем пространстве.":["Enter the server API key. Videos and results are stored in a shared workspace.","Сервердің API кілтін енгізіңіз. Бейнелер мен нәтижелер ортақ жұмыс кеңістігінде сақталады."]});
errors.invalid_api_key=['Неверный API-ключ','Invalid API key','API кілті қате'];
export function translate(text:string,language:Language):string {if(errors[text])return errors[text][language==='ru'?0:language==='en'?1:2];return language==='ru'?text:dictionary[text]?.[language==='en'?0:1]??text;}

export function formatDate(date:Date,language:Language,withTime=false){if(language==='kk'){const months=['қаң.','ақп.','нау.','сәу.','мам.','мау.','шіл.','там.','қыр.','қаз.','қар.','жел.'];return date.getDate()+' '+months[date.getMonth()]+(withTime?', '+date.getHours().toString().padStart(2,'0')+':'+date.getMinutes().toString().padStart(2,'0'):'')}return date.toLocaleString(language==='en'?'en-US':'ru-RU',{day:'numeric',month:'short',...(withTime?{hour:'2-digit',minute:'2-digit'} as const:{})});}
