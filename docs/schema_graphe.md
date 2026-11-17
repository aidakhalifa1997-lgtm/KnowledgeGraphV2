# Schéma du graphe de connaissances (v1.1)


## Entités (nœuds)

| Type | Rôle | Propriétés |
|---|---|---|
| Procedure | Pivot du graphe | reference (clé, ex. IT-02), titre, version, domaine |
| Personne | Rédacteur, vérificateur, propriétaire... | nom (clé) |
| Acteur | Fonction, rôle ou équipe qui intervient | nom (clé), nature |
| Direction | Entité de rattachement | nom (clé) |
| Domaine | Classification des procédures | nom (clé) |
| Outil | Application ou système | nom (clé) |
| Projet | Projet de transformation | code (clé), description |

Valeurs de la propriété `nature` d'un Acteur :
- poste : fonction occupée par une personne (Directeur des Systèmes d'Information)
- équipe : Service Desk, Comptabilité Fournisseurs
- comité : Comité des Changements (CAB), CODIR
- profil : rôle générique (Utilisateur, Demandeur, Collaborateur sortant)

La clé est la propriété qui identifie un nœud de façon unique. Elle sert au MERGE : si un nœud avec la même clé existe déjà dans Neo4j, il est réutilisé au lieu d'être dupliqué.

## Relations extraites par le LLM (15)

| Relation | De | Vers |
|---|---|---|
| A_POUR_PROPRIETAIRE | Procedure | Personne |
| A_POUR_REDACTEUR | Procedure | Personne |
| A_POUR_CONTRIBUTEUR | Procedure | Personne |
| A_POUR_VALIDEUR | Procedure | Personne |
| RATTACHEE_A_DIRECTION | Procedure | Direction |
| APPARTIENT_AU_DOMAINE | Procedure | Domaine |
| RATTACHEE_AU_PROJET | Procedure | Projet |
| UTILISE_OUTIL | Procedure | Outil |
| IMPLIQUE_ACTEUR | Procedure | Acteur |
| PREVOIT_CONTROLE_PAR | Procedure | Acteur |
| ALIMENTE | Procedure | Procedure |
| DEPEND_DE | Procedure | Procedure |
| COMPLETE | Procedure | Procedure |
| OCCUPE_FONCTION | Personne | Acteur (nature = poste) |
| MEMBRE_DE | Personne | Direction |

## Données hors extraction

La description d'un projet est une propriété du nœud Projet. Elle ne figure pas dans les procédures : elle est chargée depuis un référentiel de projets (code et description), hors extraction LLM, et exclue de l'évaluation. La vérité terrain compte donc 840 relations extractibles 

## Règles

1. Chaque relation porte la propriété `source` : la référence de la procédure d'où elle est extraite. Elle permet d'expliquer chaque réponse à partir du graphe.
2. Les fichiers de data/reference (relations, registre, corpus) servent uniquement à l'évaluation. Ils ne sont jamais envoyés au LLM d'extraction.
3. Résolution d'entités : normalisation (casse, accents, ponctuation), puis référentiel d'alias, avant écriture dans Neo4j. Un nom sans correspondance crée un nœud marqué "à vérifier" et n'est jamais fusionné automatiquement.
4. Le LLM reçoit la liste des noms officiels pour les types fermés (Direction, Domaine, Outil, Projet).
5. Écriture dans Neo4j avec MERGE pour permettre l'ajout de documents sans doublons.