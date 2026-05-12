from main import run_pipeline

if __name__ == "__main__":
    for mode in ['own', 'rent']:
        print(f"\nОбучение модели для режима: {mode}")
        run_pipeline(mode=mode, save_artifacts=True)
    print("\nВсе артефакты сохранены в models/")