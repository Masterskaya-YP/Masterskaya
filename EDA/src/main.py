from .chat_analyzer import ChatAnalyzer

def main():
    print("=== Анализатор чатов ===")
    json_path = input("Введите путь к JSON-файлу с данными чата: ").strip('"')
    
    if not os.path.exists(json_path):
        print(f"Ошибка: файл не найден - {json_path}")
        return
    
    analyzer = ChatAnalyzer()
    if analyzer.analyze_chat(json_path):
        print("Анализ успешно завершен! Проверьте папку 'output'")
    else:
        print("Произошла ошибка при анализе. Проверьте логи для деталей.")

if __name__ == "__main__":
    import os
    main()