def test_list_models(client):
    models = client.models.list()
    model_names = [model.id for model in models.data]
    assert "mlx-community/Qwen3.5-4B-MLX-4bit" in model_names
    assert "z_image.ZImage/z_image_turbo" in model_names
