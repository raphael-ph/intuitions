---
title: "A imaginação é mais importante que o conhecimento"
subtitle: "Minhas intuições sobre o artigo Mastering Atari with Discrete World Models"
description: "Minha intuição pessoal do DreamerV2"
pubDate: 2026-07-05
# updatedDate: 2026-06-18
draft: true
slug: "87082f67-2a38-4047-b528-a596122c71a8"
---
<figure>

![Interação de Pong](/assets/dreamerv2/pong-interaction.gif)
<figcaption>Representação do jogo Pong (gerada pelo autor)</figcaption>
</figure>

*Pong* foi o primeiro jogo desenvolvido pela Atari, lançado em 1972. Curiosamente, ele nasceu como um exercício para o engenheiro Allan Alcorn. Os cofundadores da Atari ficaram tão impressionados com a qualidade do protótipo que decidiram empacotá-lo e distribuí-lo comercialmente[^1]. O resultado foi um sucesso fenomenal, afinal, mais de 50 anos depois, ele ainda é um jogo amplamente conhecido e recriado.

As regras são bastante simples, cada jogador controla uma raquete que desliza verticalmente na tela. Controlando sua raquete, o jogador deve impedir a bolinha de chegar até o seu lado do monitor, rebatendo-a para o outro lado, sendo possível se aproveitar das bordas superior e inferior para "tabelar" a jogada. Ganha aquele que fechar 21 pontos primeiro[^2].

Apesar da premissa descomplicada, o jogo pode se tornar bastante complexo, mesmo jogando contra a máquina (desafio o leitor a gastar alguns minutos para vencer o nível *hard* neste [site](https://www.ponggame.org/)). Um detalhe importante, no primeiro lançamento, Pong não tinha qualquer tipo de inteligência artificial para a versão *single player*. A raquete controlada pela máquina posiciona-se de modo a igualar a altura da bolinha no monitor, sendo os níveis mais difíceis definidos simplesmente pela velocidade da raquete.

Tornar-se um jogador de Pong de habilidades invejáveis exige o conhecimento da **dinâmica** do ambiente. Veja se o leitor seria capaz de adivinhar para qual posição a bolinha seria lançada neste cenário aqui:

![Previsão do Pong](/assets/dreamerv2/pong-prediction.gif)

A escolha da posição correta depende bastante do **modelo mental** que temos do jogo. Quando entendemos seu funcionamento, somos capazes de prever com precisão a posição futura de um lançamento que ricocheteou da raquete adversária para cima e para baixo antes de aterrissar do outro lado. Quase como se pudéssemos ver o jogo acontecendo em nossas cabeças, antecipando as jogadas do oponente.

Em 2021, um grupo de pesquisadores apresentou na ICLR (*International Conference on Learning Representations*) um artigo que explora a hipótese do aprendizado em um espaço compacto ou **espaço latente**, isto é, seria possível desenvolver um "modelo mental" de determinado ambiente, suficientemente preciso, que nos permitisse treinar um agente sobre este ambiente e então transportar suas ações com sucesso para o ambiente real, sem perda de generalidade? Os autores introduziram o Agente Inteligente DreamerV2[^3], uma evolução do DreamerV1[^4], que apresentou resultados interessantes, superando, em muito, o desempenho de jogadores humanos em diversos *benchmarks* com jogos de Atari.

Este artigo me foi apresentado em 2025 durante uma aula de mestrado e fiquei fascinado com seu conceito. Recentemente, repliquei a arquitetura apresentada no *paper*, limitando-me a treinar um agente capaz de dominar exclusivamente Pong. No artigo original, os autores generalizam a implementação para 36 jogos de Atari diferentes. Minha implementação, em PyTorch, pode ser encontrada neste [repositório](https://github.com/raphael-ph/pong-dreamer). A implementação original, em JAX, pode ser encontrada neste [repositório](https://github.com/danijar/dreamerv2).

Em linhas gerais, o processo de aprendizado do DreamerV2 pode ser dividido em dois blocos, sendo o primeiro o treinamento de um *World Model* e o segundo, referente ao treinamento de um **Ator** e um **Crítico**. É fundamental destacar que o treinamento de ambos os blocos ocorre de forma paralela, isto é, constrói-se um modelo que seja capaz de simular completamente o ambiente e, simultaneamente, utiliza-se este modelo para gerar ou, nas palavras dos autores, "sonhar" novos cenários, a partir de uma única entrada. Em seguida, treinamos um agente "jogador", que aprenderá a jogar sem que sejam necessárias imagens reais, apenas os sonhos.

## Treinamento do *World Model*

O primeiro passo é entender o que é *World Model*. Recentemente, a Dra. Fei-Fei Li (World Labs[^5]) escreveu um post em seu blog[^6] onde trouxe uma definição para os modelos de mundo. A professora traz uma definição interessante:

> Where language models learn the statistical structure of text, world models learn the statistical structure of space and time: how light falls on a surface, how a garden looks from an angle no camera has captured, how objects respond to force and follow the laws of physics.

e a complementa com uma descrição funcional de um *World Model*. Segundo ela, um modelo de mundo deve ter todas ou algumas das seguintes capacidades:

* *Renderer*: Tem como saída imagens, vídeos, pixels, que podem ser interpretados pelos olhos das pessoas;
* *Simulator*: Tem como saída um **estado** uma representação geométrica, física e dinamicamente coerente com a representação do ambiente em questão, podendo ser interpretada tanto por pessoas quanto como por máquinas;
* *Planner*: Tem como saída ações, isto é, dada uma observação, projeta qual a melhor ação a ser tomada.

Considerando a taxonomia proposta, pode-se enquadrar o Dreamer em um modelo que combine o Simulador e Planejador, visto que nosso objetivo é recriar uma visão simulada do jogo e planejar quais serão os melhores movimentos de modo a maximizar o nosso ganho, no caso do Pong, buscar os 21 pontos vencedores. Essencialmente, queremos construir um sistema que seja capaz de:

- Reproduzir corretamente as imagens, ou seja, reconstruir o ambiente de modo a se visualmente coerente com o cenário real.
- Reproduzir corretamente as transições, isto é, dada uma entrada, os próximos passos a partir dela deve ser coerente e fazer sentido com a dinâmica do ambiente real;
- Reproduzir corretamente a recompensa por aquele episódio, isto é, no caso do Pong, pontuar corretamente dependendo de qual jogador foi o vencedor do rally.

<figure>

![Treinamento do Modelo de Mundo](/assets/dreamerv2/wm-training.gif)
<figcaption>Intuição sobre o treinamento do World Model do DreamerV2</figcaption>

</figure>

O treinamento começa a partir de imagens reais do jogo. A entrada do modelo é composta por um par *frame* e **ação**, ou seja, para cada imagem, dizemos ao modelo o que os jogadores estavam fazendo. Esse par passa então por aquilo que chamamos de Redes Neurais Convolucionais (do inglês, *Convolutional Neural Network* ou **CNN** daqui em diante). O objetivo desta rede é **COMPRIMIR** as informações contidas na imagem em uma versão compactada, que concentre apenas as informações cruciais que aquela imagem nos comunica. Nela temos, por exemplo, a localização do placar, o tamanho das raquetes e o tamanho da bolinha. A esta representação, damos o nome de *embedding*.

<figure>

![Encoder](/assets/dreamerv2/encoder.gif)
<figcaption>Demonstração de como o modelo compacta informações da imagem</figcaption>

</figure>

Além da representação visual do jogo, é necessário trazer informações também de sua dinâmica como, por exemplo, a posição da raquete do adversário, onde a bolinha acertou a raquete do adversário ou se a bolinha ricocheteou de alguma das bordas. Estas informações também precisam ser representadas, e os autores encontraram uma forma interessante de fazê-la. Em essência, forçaram o modelo de mundo a compactar estas informações em uma grande tabela, com 32 colunas e 32 linhas, onde cada linha representa uma dinâmica do jogo e cada coluna um possível estado daquela dinâmica. Assim, a primeira linha pode representar a dinâmica da raquete adversária, e cada uma das colunas representa sua posição na tela; a segunda linha pode, por sua vez, representar a dinâmica de posicionamento da bolinha e assim sucessivamente. O modelo precisa **compactar** essa informação em vetores com valores 0 ou 1 de modo que essa classificação seja emergente do treinamento do modelo, isto é, ele próprio aprende a escolher o que cada uma das linhas e colunas deverá representar. A escolha por usar apenas valores inteiros (dinâmica categórica) representa uma grande descoberta. Os autores demonstram que os resultados são superiores a representações com valores quebrados (dinâmica gaussiana)[^7].

<figure>

![Memory](/assets/dreamerv2/memory.gif)
<figcaption>Demonstração de como o modelo compacta informações da dinâmica do ambiente</figcaption>

</figure>

No entanto, imagens isoladas são insuficientes para determinar com precisão o desenrolar do jogo. Veja a imagem a seguir por exemplo:

<figure>

![Imagem isolada de um rally de Pong](/assets/dreamerv2/pong_shot.png)
<figcaption>Imagem isolada de um rally de Pong</figcaption>

</figure>

Sem ter ao menos uma única imagem do segundo anterior, fica muito difícil saber de que lado a bolinha está vindo. No DreamerV2, os autores utilizaram um método bastante interessante para resolver este problema. Eles utilizam aquilo que chamam de *Recurrent State-Space Machine*, **RSSM** daqui em diante. Essa estrutura funciona, essencialmente, como uma grande memória, também em espaço latente, que representa os estados anteriores do jogo. Nela temos informações como de onde a bolinha veio, onde ela bateu por último e onde as raquetes estavam.

Consolidando todas essas informações, podemos ensinar o modelo a simular o ambiente:

<figure>

![Dreaming](/assets/dreamerv2/dreaming.gif)
<figcaption>Demonstração de como ocorre o processo de simulação</figcaption>

</figure>

## Treinamento do Ator e do Crítico

Agora que temos um simulador, o próximo passo é usá-lo para, efetivamente, ensinar um agente a jogar o jogo. Neste caso, utilizaremos a técnica de **Ator e Crítico**. Essencialmente, teremos um modelo responsável por jogar o jogo no ambiente simulado, e outro que tentará adivinhar quantos pontos o jogador fará de acordo com a forma como ele está jogando.

<figure>

![A2C](/assets/dreamerv2/a2c.gif)
<figcaption>Representação do processo de aprendizado do Ator e Crítico utilizando a representação compactada do ambiente.</figcaption>

</figure>

Essa técnica é bastante clássica (caso queira se aprofundar, recomendo o Capítulo 13.5 de [^8]), a grande novidade está no fato de que, com esta implementação, não é necessário utilizar jogadas reais, mas sim um ambiente totalmente simulado. Isto é, o agente que está sendo treinado não se utiliza dos píxeis em tela do ambiente verdadeiro, mas sim da imagem compactada (latente) que foi desenvolvida pelo modelo de mundo.
 
## Resultados

Coloco aqui agora alguns resultados reais que tive ao desenvolver este modelo. Durante o processo de treinamento, pode-se ver a evolução da aprendizagem e como o modelo começa a capturar fatores reais da dinâmica do ambiente.

Vemos que no começo, praticamente não existem raquetes e a pontuação é quase um borrão, ao passo que, eventualmente, o modelo passa a representar corretamente o ambiente:

<figure>

![Imagem da reconstrução do World Models](/assets/dreamerv2/reconstruction_compare.png)
<figcaption>Comparação entre a capacidade de reconstrução do início do treinamento (à esquerda) e do final do treinamento (à direita). As imagens acima representam a cena real extraída do ambiente que o modelo está tentando reconstruir.</figcaption>

</figure>

E durante a simulação, observa-se que no início, o "sonho" do modelo de mundo é, praticamente um borrão. Conforme seu treinamento passa a evoluir, o modelo começa a aprender toda a dinâmica do ambiente. Nos vídeos abaixo, pode-se ver inclusive a dinâmica de pontuação sendo corretamente representada no ambiente:

<!-- Simulação 10k passos -->
<figure style="max-width: 1000px; width: 50%; margin-inline: auto;">

![Simulação 10k passos](/assets/dreamerv2/imagination_rollout_step0010000.gif)
<figcaption>Ambiente simulado pelo modelo de mundo no início do treinamento.</figcaption>

</figure>

<!-- Simulação 1M de Passos -->
<figure style="max-width: 1000px; width: 50%; margin-inline: auto;">

![Simulação 1M passos](/assets/dreamerv2/imagination_rollout_step0950000.gif)
<figcaption>Ambiente simulado pelo modelo de mundo ao final do treinamento.</figcaption>

</figure>

E, claro, não poderia faltar o resultado final, do agente treinado, tornando-se um excelente jogador de Pong:

<figure style="max-width: 1000px; width: 50%; margin-inline: auto;">

<video src="/assets/dreamerv2/episode_007.mp4" autoplay loop muted playsinline controls></video>
<figcaption>Ambiente simulado pelo modelo de mundo ao final do treinamento. O agente treinado é representado pela raquete à esquerda.</figcaption>

</figure>

É interessante observar como o agente aprendeu a explotar uma característica do Pong: acertar a bolinha com a quina da raquete faz com que ela acelere bastante, tornando-se bem mais difícil de ser recepcionada pelo oponente.

## Conclusão

Recriar o DreamerV2 em PyTorch e vê-lo dominar o Pong foi um exercício bastante interessante e me permitiu colocar a frase de Einstein, que dá título a este post, à prova da engenharia. No aprendizado por reforço tradicional, o agente é um acumulador de conhecimento bruto: ele precisa errar e bater na parede milhões de vezes no mundo real para entender o ambiente. No caso do DreamerV2, o agente aprendeu a explotar as quinas da raquete sem sequer interagir uma vez no ambiente real.

> World models are how machines will finally come to understand, imagine, reason and interact with it.[^6]

## REFERÊNCIAS

[^1]: [Pong](https://en.wikipedia.org/wiki/Pong)
[^2]: [Video Olympics - Atari - Atari 2600](https://atariage.com/manual_html_page.php?SoftwareLabelID=587)
[^3]: [Mastering Atari with Discrete World Models](http://arxiv.org/abs/2010.02193)
[^4]: [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/pdf/1912.01603)
[^5]: [World Labs](https://www.worldlabs.ai/)
[^6]: [A Functional Taxonomy of World Models](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models)
[^7]: [Google Research Blog](https://research.google/blog/mastering-atari-with-discrete-world-models/)
[^8]: [Reinforcement learning: an introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)