# Acesso necessario para rodar no Colab

Eu nao preciso da sua senha do Google, GitHub ou Kaggle.

Para voce rodar tudo na sua conta, precisa apenas de:

1. Acesso ao Google Colab com GPU T4 ativada.
2. O repositorio no GitHub ou a pasta do projeto enviada para o Colab.
3. Opcionalmente, um token Kaggle `kaggle.json`.

## Como gerar o `kaggle.json`

1. Entre no Kaggle.
2. Clique na foto de perfil.
3. Acesse `Account`.
4. Va ate a secao `API`.
5. Clique em `Create New Token`.
6. O Kaggle baixara um arquivo chamado `kaggle.json`.

Esse arquivo deve ser enviado manualmente no notebook quando a celula pedir upload. Nao suba esse arquivo no GitHub.

## Quando o token e necessario?

O script usa `kagglehub`, que muitas vezes baixa datasets publicos sem autenticacao. Se o download falhar por permissao/autenticacao, use o upload do `kaggle.json`.

## O que nao compartilhar

- Senha do Google.
- Senha do Kaggle.
- `kaggle.json` no GitHub.
- Tokens pessoais por chat.

## Comando principal no Colab

```bash
pip install -r requirements-colab.txt
python src/download_data.py
python src/train_evaluate_gpu_colab.py --folds 5
```
