# Zettelkasten プロジェクト タグ付け規則

## 概要

このプロジェクトでは、Austin Govellaの原則 **"Tags provide architecture, not content"**（タグは内容ではなく構造を提供する）に基づいて、階層的なタグ体系を採用しています。

## タグ体系の基本構造

タグは **カテゴリ/詳細** の形式で記述し、以下の4つの主要カテゴリに分類されます：

### 1. 📝 Type分類（ノートタイプ）
```
#type/fleeting     - フリーティングノート（一時的なアイデア）
#type/literature   - 文献ノート（読書・研究記録）
#type/permanent    - 永続ノート（完成された洞察）
#type/project      - プロジェクトノート（具体的なプロジェクト）
#type/structure    - 構造ノート（索引・MOC・ナビゲーション）
#type/note         - 一般的なノート
#type/sketchnote   - スケッチノート
#type/book         - 書籍記録
#type/quote        - 引用記録
#type/term         - 用語定義
#type/person       - 人物記録
#type/tool         - ツール評価
#type/meeting      - 会議記録
#type/post         - ブログ投稿
#type/question     - 質問記録
#type/prompt       - プロンプト記録
#type/recipe       - レシピ
#type/okr          - 目標設定
#type/rule         - ルール・原則
#type/example      - 事例・例
#type/dataview     - DataViewクエリ
```

### 2. 🏗️ Structure分類（構造タイプ）
```
#structure/moc     - Map of Content（内容地図）
#structure/index   - インデックス
#structure/about   - 説明ページ
#structure/canvas  - キャンバス
#structure/list    - リスト
```

### 3. 🎯 Theme分類（テーマ・主題）
```
#theme/zettelkasten    - Zettelkasten方法論
#theme/productivity    - 生産性
#theme/learning        - 学習
#theme/research        - 研究
#theme/pkm             - Personal Knowledge Management
#theme/sketchnotes     - スケッチノート
#theme/writing         - ライティング
#theme/thinking        - 思考
#theme/datastory       - データストーリー
#theme/cooking         - 料理
#theme/finance         - 金融
#theme/objectives      - 目標管理
```

### 4. 🎯 Target分類（対象・用途）
```
#target/starterkit         - スターターキット用
#target/project            - プロジェクト関連
#target/github             - GitHub関連
#target/linkedin           - LinkedIn投稿用
#target/forumzettelkasten  - Zettelkastenフォーラム用
#target/forumobsidian      - Obsidianフォーラム用
#target/reddit             - Reddit投稿用
```

### 5. 🎭 Role分類（役割・専門分野）
```
#role/author           - 著者
#role/expert           - 専門家
#role/visualthinker    - ビジュアル思考者
#role/networkedthinker - ネットワーク思考者
```

### 6. 📊 Status分類（状態管理）
```
#status/open       - 進行中
#status/wip        - 作業中
#status/backlog    - バックログ
#status/active     - アクティブ
```

### 7. 📖 Source分類（情報源）
```
#source/chatgpt    - ChatGPT由来
```

### 8. 📈 Diagram分類（図表タイプ）
```
#diagram/doublebubble-map  - ダブルバブルマップ
```

## 使用ルール

### ✅ 必須タグ
すべてのノートには最低限以下のタグを付与：
- **1つの`#type/`タグ** - ノートの種類を特定
- **1つ以上の`#theme/`タグ** - 主題を特定

### 📋 推奨タグ
- **`#target/`タグ** - 特定の用途がある場合
- **`#structure/`タグ** - 構造ノートの場合

### 🔄 DataViewクエリでの活用
```dataview
FROM #type/book
FROM #theme/zettelkasten
FROM #status/open OR #status/wip
FROM #target/forumzettelkasten
```

### 🏷️ ソーシャルメディア用タグ
LinkedInなどの投稿では、通常のタグとは別に公開用ハッシュタグも管理：
```
#edmund2024 #obsidian #zettelkasten #pkm #knowledgemanagement
```

## 命名規則

1. **小文字使用** - すべて小文字で記述
2. **階層構造** - カテゴリ/詳細の形式
3. **英語使用** - 国際的な互換性のため
4. **簡潔性** - 短く分かりやすい名前
5. **一貫性** - 同じ概念には同じタグを使用

## ベストプラクティス

### ✨ 効果的なタグ付け
- **過度なタグ付けを避ける** - 5-7個程度に抑制
- **意味のあるタグのみ使用** - 検索・フィルタリングに有用なもの
- **継承性を意識** - 上位概念から下位概念へ
- **進化を許容** - 必要に応じてタグ体系を更新

### 🔍 検索とフィルタリング
- Obsidianのタグペインで階層表示
- DataViewクエリでの動的リスト生成
- 複数タグでのAND/OR検索

## 参考文献

- "Tags provide architecture, not content" - Austin Govella
- [How to use tags in a PKM like Obsidian](https://austingovella.medium.com/how-to-approach-tags-in-your-pkm-b29c98dc43d3) 