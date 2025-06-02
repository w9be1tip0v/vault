# Obsidian Zettelkasten Knowledge Base Configuration Guide

> [!info] Overview
> A complete configuration guide for implementing an Obsidian knowledge base based on Niklas Luhmann's principles and Sönke Ahrens' "How to Take Smart Notes" methodology

## 🏗️ Folder Structure

### 1️⃣ Fleeting Notes (1_Fleeting)
```
1_Fleeting/
├── About - 1_Fleeting Notes.md
└── Daily ideas and memos
```

**Purpose**: Quickly record temporary ideas and inspiration
**Rules**:
- Record ideas as they come
- Regularly review and promote valuable ones to 2_Literature or 3_Permanent
- Process within short timeframes (review 1-2 times per week)

### 2️⃣ Literature Notes (2_Literature)
```
2_Literature/
├── _About - 2_Literature Notes.md
├── [Book Title] - [Author Name].md
├── [Person Name].md
├── [Concept/Term].md
└── Tags provide architecture.md
```

**Purpose**: Organize and summarize knowledge from external sources
**Rules**:
- Clearly cite information sources
- Summarize in your own words
- Record important quotes accurately
- Create notes per source

### 3️⃣ Permanent Notes (3_Permanent)
```
3_Permanent/
├── _About - 3_Permanent Notes.md
├── [Independent Idea].md
└── Basic Zettelkasten vs Extended Digital Zettelkasten.md
```

**Purpose**: Fully processed, independent ideas and insights
**Rules**:
- One idea, one note
- Write in your own words
- Create meaningful links to other notes
- Ensure future self can understand

### 4️⃣ Project Notes (4_Project)
```
4_Project/
├── About - 4_Project Notes.md
├── [Project Name]/
│   ├── _MOC - [Project Name].md
│   ├── Project Log - [Project Name].md
│   └── History - [Project Name].md
└── Posts Zettelkasten/
```

**Purpose**: Information related to specific time-bound projects
**Rules**:
- Create folder per project
- Use MOC (Map of Content) to oversee everything
- Archive after completion

### 5️⃣ Structure Notes (5_Structure)
```
5_Structure/
├── _About - 5_Structure Notes.md
├── Index - Zettelkasten.md
├── Terms - Zettelkasten.md
├── Glossary.md
├── Books.md
├── Quotes.md
├── ARCO View.md
├── Inspect View.md
├── 7 Days Created Chart.md
└── Canvases/
    ├── About - 1_Fleeting Notes.canvas
    ├── About - 2_Literature Notes.canvas
    ├── About - 3_Permanent Notes.canvas
    ├── About - 4_Project Notes.canvas
    └── About - 5_Structure Notes.canvas
```

**Purpose**: System-wide navigation and organization of related notes
**Rules**:
- Function as hubs for related notes
- Update regularly
- Generate dynamically with Dataview queries

## 🏷️ Tag System

### Type Classification
- `#type/fleeting` - Fleeting notes
- `#type/literature` - Literature notes  
- `#type/permanent` - Permanent notes
- `#type/project` - Project notes
- `#type/structure` - Structure notes

### Structure Classification
- `#structure/moc` - Map of Content
- `#structure/index` - Index
- `#structure/about` - About pages
- `#structure/canvas` - Canvas
- `#structure/list` - List

### Theme Classification
- `#theme/zettelkasten` - Zettelkasten related
- `#theme/productivity` - Productivity
- `#theme/learning` - Learning
- `#theme/research` - Research

### Target Classification
- `#target/project` - Project related
- `#target/github` - GitHub related
- `#target/starterkit` - Starter kit

## 📝 Template Configuration

### Basic Templates
- **2_Book Template** - For book reviews
- **2_Person Template** - For person records
- **2_Quote Template** - For quote records
- **2_Term Template** - For term definitions
- **2_Tool Template** - For tool evaluations
- **3_Note Template** - For permanent notes
- **3_Question Template** - For question records
- **5_Structure Template** - For structure notes

### Project Templates
- **4_Post Template** - For blog posts
- **4_E-book Template** - For e-books
- **5_Meeting Notes Template** - For meeting records
- **5_OKR Template** - For goal management

### Productivity Templates
- **5_BuJo - Daily Log** - Daily log
- **5_BuJo - Weekly Log** - Weekly log
- **5_BuJo - Monthly Log** - Monthly log
- **5_BuJo - Future Log** - Future log

## 🔗 Linking Strategy

### Internal Links
```markdown
[[Note Name]]           # Basic link
[[Note Name|Display]]   # Alias link
[[Note Name#Heading]]   # Section link
![[Note Name]]          # Embed link
```

### Backlink Utilization
- Automatic discovery of related notes
- Tracking idea evolution
- Knowledge clustering

### MOC Structuring
- Aggregation of related topics
- Navigation hubs
- Knowledge map creation

## 📊 Dataview Utilization

### Dynamic Query Example
```dataview
TABLE WITHOUT ID 
	file.link as "Recent Notes", 
	(date(today) - file.cday).day as "Days Old",
	template_type as "Type"
FROM "3_Permanent" 
WHERE template_type
SORT file.cday desc 
LIMIT 10
```

### Homepage Features
- Random quote display
- Recent activity display
- Statistics display

## 🔄 Workflow

### Daily Process
1. **New Ideas** → Record in `1_Fleeting`
2. **Reading & Research** → Summarize in `2_Literature`
3. **Idea Processing** → Promote valuable ones to `3_Permanent`

### Weekly Review
1. Organize and process `1_Fleeting`
2. Strengthen links
3. Discover and connect isolated notes

### Monthly Maintenance
1. Review structure
2. Organize tags
3. Check project progress

## ⚙️ Recommended Plugins

### Essential Plugins
- **Dataview** - Dynamic queries
- **Templates** - Template management
- **Obsidian Git** - Version control

### Recommended Plugins
- **Graph Analysis** - Graph analysis
- **Tag Wrangler** - Tag management
- **Calendar** - Chronological navigation
- **Kanban** - Project management

## 📈 Growth Strategy

### Starting Period (0-3 months)
- Understanding basic structure
- Learning templates
- Establishing daily habits

### Development Period (3-12 months)  
- Expanding knowledge network
- Deepening specialized areas
- Personal customization

### Maturity Period (12+ months)
- System optimization
- Advanced query utilization
- External knowledge sharing

---

> [!success] Key to Success
> **Consistency** > Perfection. Start small and gradually grow your system.

**References**:
- Sönke Ahrens "How to Take Smart Notes"
- [Obsidian-Zettelkasten-Starter-Kit](https://github.com/groepl/Obsidian-Zettelkasten-Starter-Kit)
- [Obsidian Official Documentation](https://obsidian.md) 