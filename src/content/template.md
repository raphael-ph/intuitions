---
title: "Hello World"
subtitle: "Oi"
description: "One-line summary shown in meta tags."
pubDate: 2026-06-18
# updatedDate: 2026-06-18
---

Primeiro Post!

<!--
Tudo abaixo é um cheat sheet de blocos prontos pra copiar/colar.
Apague o que não for usar antes de publicar.
-->

## Heading 2

### Heading 3

#### Heading 4

Parágrafo normal com **negrito**, *itálico* e `código inline`.

- item de lista
- outro item

1. item numerado
2. outro item

> Blockquote / citação em destaque.

[link interno](/about) · [link externo](https://example.com)

---

### Imagem

<!-- arquivo em public/images/<post>/nome.png -->
![Texto alternativo](/images/dreamerv2/nome.png)

### GIF

<!-- mesmo esquema, arquivo em public/videos/<post>/nome.gif — o .gif no final do src ativa o destaque dourado -->
<img src="/videos/dreamerv2/nome.gif" alt="Texto alternativo" />

### Vídeo

<!-- arquivo em public/videos/<post>/nome.mp4 -->
<video src="/videos/dreamerv2/nome.mp4" controls playsinline></video>

### Código

```python
def hello():
    print("oi")
```

### Tabela

| Coluna A | Coluna B |
| -------- | -------- |
| valor 1  | valor 2  |

### LaTeX

<!-- inline: $...$ · bloco: $$...$$ em linha própria, com linha em branco antes e depois -->
Fórmula inline $E = mc^2$ no meio do texto.

$$\mathcal{L}(\phi) \doteq \mathbb{E}_{q_\phi(z \mid x)} \big[ -\ln p_\phi(x \mid z) + \text{KL}\big[ q_\phi(z \mid x) \parallel p(z) \big] \big]$$

### Referência (nota de rodapé)

<!-- o número referencia o texto no fim do post; pode repetir [^1] em vários lugares -->
Alguma afirmação que precisa de fonte[^1].

[^1]: Nome do autor, "Título do artigo", ano. [link](https://example.com)

### Legenda / subtítulo (funciona em qualquer bloco acima)

<!--
Envolva qualquer bloco (imagem, gif, vídeo, código, tabela) num <figure>,
com linha em branco antes e depois do conteúdo, e feche com <figcaption>.
-->

<figure>

![Texto alternativo](/images/dreamerv2/nome.png)

<figcaption>Legenda da imagem aqui.</figcaption>
</figure>

<figure>

```python
def hello():
    print("oi")
```

<figcaption>Legenda do código aqui.</figcaption>
</figure>

<figure>

| Coluna A | Coluna B |
| -------- | -------- |
| valor 1  | valor 2  |

<figcaption>Legenda da tabela aqui.</figcaption>
</figure>
