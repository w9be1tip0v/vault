# Zettelkasten Project Tagging Rules

## Overview

This project adopts a hierarchical tagging system based on Austin Govella's principle: **"Tags provide architecture, not content"** (tags provide structure, not content).

## Basic Tag System Structure

Tags are written in **Category/Detail** format and classified into the following 9 major categories:

### 1. 📝 Type Classification (Note Types)
```
#type/fleeting     - Fleeting notes (temporary ideas)
#type/literature   - Literature notes (reading & research records)
#type/permanent    - Permanent notes (completed insights)
#type/project      - Project notes (specific projects)
#type/structure    - Structure notes (indexes, MOCs, navigation)
#type/note         - General notes
#type/sketchnote   - Sketch notes
#type/book         - Book records
#type/quote        - Quote records
#type/term         - Term definitions
#type/person       - Person records
#type/tool         - Tool evaluations
#type/meeting      - Meeting records
#type/post         - Blog posts
#type/question     - Question records
#type/prompt       - Prompt records
#type/recipe       - Recipes
#type/okr          - Objective setting
#type/rule         - Rules & principles
#type/example      - Cases & examples
#type/dataview     - DataView queries
```

### 2. 🏗️ Structure Classification (Structure Types)
```
#structure/moc     - Map of Content
#structure/index   - Index
#structure/about   - About pages
#structure/canvas  - Canvas
#structure/list    - List
```

### 3. 🎯 Theme Classification (Themes & Topics)
```
#theme/zettelkasten    - Zettelkasten methodology
#theme/productivity    - Productivity
#theme/learning        - Learning
#theme/research        - Research
#theme/pkm             - Personal Knowledge Management
#theme/sketchnotes     - Sketch notes
#theme/writing         - Writing
#theme/thinking        - Thinking
#theme/datastory       - Data storytelling
#theme/cooking         - Cooking
#theme/finance         - Finance
#theme/objectives      - Objective management
```

### 4. 🎯 Target Classification (Target & Purpose)
```
#target/starterkit         - For starter kit
#target/project            - Project related
#target/github             - GitHub related
#target/linkedin           - For LinkedIn posts
#target/forumzettelkasten  - For Zettelkasten forum
#target/forumobsidian      - For Obsidian forum
#target/reddit             - For Reddit posts
```

### 5. 🎭 Role Classification (Roles & Expertise)
```
#role/author           - Author
#role/expert           - Expert
#role/visualthinker    - Visual thinker
#role/networkedthinker - Networked thinker
```

### 6. 📊 Status Classification (Status Management)
```
#status/open       - In progress
#status/wip        - Work in progress
#status/backlog    - Backlog
#status/active     - Active
```

### 7. 📖 Source Classification (Information Sources)
```
#source/chatgpt    - ChatGPT derived
```

### 8. 📈 Diagram Classification (Diagram Types)
```
#diagram/doublebubble-map  - Double bubble map
```

### 9. 🔑 Keyword Classification (Keywords & Concepts)
```
#keyword/transformer       - Transformer architecture
#keyword/attention         - Attention mechanism
#keyword/machine-learning  - Machine learning
#keyword/nlp              - Natural Language Processing
#keyword/factuality       - Factuality evaluation
#keyword/grounding        - Text grounding
#keyword/evaluation       - Model evaluation
#keyword/benchmark        - Benchmarking
```

## Usage Rules

### ✅ Required Tags
All notes must include at minimum:
- **One `#type/` tag** - Identify the note type
- **One or more `#theme/` tags** - Identify the subject matter

### 📋 Recommended Tags
- **`#target/` tag** - When there's a specific purpose
- **`#structure/` tag** - For structure notes
- **`#keyword/` tags** - For technical terms and concepts

### 🔄 DataView Query Integration
```dataview
FROM #type/book
FROM #theme/zettelkasten
FROM #status/open OR #status/wip
FROM #target/forumzettelkasten
FROM #keyword/machine-learning
```

### 🏷️ Social Media Tags
For posts on LinkedIn and other platforms, manage separate public hashtags alongside regular tags:
```
#edmund2024 #obsidian #zettelkasten #pkm #knowledgemanagement
```

## Naming Conventions

1. **Use lowercase** - Write everything in lowercase
2. **Hierarchical structure** - Category/Detail format
3. **Use English** - For international compatibility
4. **Conciseness** - Short and clear names
5. **Consistency** - Use same tags for same concepts

## Best Practices

### ✨ Effective Tagging
- **Avoid over-tagging** - Keep to around 5-7 tags
- **Use meaningful tags only** - Useful for search & filtering
- **Consider inheritance** - From general to specific concepts
- **Allow evolution** - Update tag system as needed

### 🔍 Search and Filtering
- Hierarchical display in Obsidian's tag pane
- Dynamic list generation with DataView queries
- AND/OR search with multiple tags

## References

- "Tags provide architecture, not content" - Austin Govella
- [How to use tags in a PKM like Obsidian](https://austingovella.medium.com/how-to-approach-tags-in-your-pkm-b29c98dc43d3) 