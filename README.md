# vwm_rnn
Biologically plausible rnn for visual working memory tasks

## Dependency Hierarchy
All files depend on config.py

```mermaid
graph TD
    config[config.py] --> rnn
    rnn[rnn.py] --> utils[utils.py]
    utils --> train[train.py]
    train --> main[main.py]
