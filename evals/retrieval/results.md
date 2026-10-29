# Retrieval evaluation

- **Run:** 2026-10-29 16:35 UTC
- **Embedding model:** `BAAI/bge-small-en-v1.5`
- **Query prefix:** `Represent this sentence for searching relevant passages: `
- **Index:** 21 sections from docs/policies/
- **Questions:** 30 (questions.yaml)
- **Fusion:** RRF, k = 60
- **Latency:** per query, in process: query embedding (vector and hybrid) and SQL

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Mean latency | p95 latency |
|---|---|---|---|---|---|---|
| Vector only | 0.600 | 0.900 | 1.000 | 0.764 | 11.3 ms | 12.7 ms |
| Full-text only | 0.400 | 0.633 | 0.700 | 0.552 | 1.3 ms | 1.9 ms |
| Hybrid (RRF) | 0.633 | 0.767 | 0.900 | 0.741 | 13.1 ms | 15.5 ms |

Rank of the correct section per question (- = not returned):

| Id | Question | Expected section | Vector | Full-text | Hybrid |
|---|---|---|---|---|---|
| d1 | My blender showed up with a cracked jug. Who do I tell, and how soon? | `damaged-items#what-should-i-do-if-an-item-arrives-damaged-or-isnt-what-i-ordered` | 1 | - | 2 |
| d2 | What information will you need from me about a defective product? | `damaged-items#what-should-i-do-if-an-item-arrives-damaged-or-isnt-what-i-ordered` | 4 | - | 6 |
| d3 | You sent me a blue kurta instead of the green one I ordered. What are my options? | `damaged-items#what-can-i-get` | 3 | 6 | 2 |
| d4 | Does a replacement for a damaged item cover my whole order or just that item? | `damaged-items#what-can-i-get` | 1 | 1 | 1 |
| d5 | The lamp arrived broken six weeks ago. Can I still get it replaced? | `damaged-items#is-there-a-time-limit` | 2 | 9 | 5 |
| d6 | Is there a maximum value for replacing a damaged TV? | `damaged-items#are-there-limits` | 1 | 6 | 4 |
| d7 | I already got my money back for the broken headphones. Can you also send new ones? | `damaged-items#what-if-the-item-has-already-been-refunded` | 1 | 1 | 1 |
| r1 | In which situations do you give money back? | `refunds#when-can-i-get-a-refund` | 4 | - | 6 |
| r2 | Can I be refunded twice for the same thing? | `refunds#when-can-i-get-a-refund` | 2 | 1 | 1 |
| r3 | How many days until the refund shows up in my bank account? | `refunds#how-will-i-be-refunded-and-when` | 1 | 2 | 1 |
| r4 | Will the money go back to my card or come as store credit? | `refunds#how-will-i-be-refunded-and-when` | 1 | 1 | 1 |
| r5 | What's the most the assistant can refund on its own for one order? | `refunds#are-there-limits-on-refunds` | 1 | 1 | 1 |
| r6 | I've had several refunds this month. Is there a cap per customer? | `refunds#are-there-limits-on-refunds` | 1 | 1 | 1 |
| r7 | Can you give me some compensation as an apology for all the trouble? | `refunds#goodwill-refunds` | 3 | 1 | 1 |
| r8 | I cancelled before it shipped. Do I need to ask for my money back? | `refunds#cancelled-orders` | 1 | 1 | 1 |
| t1 | How many days do I have to send something back? | `returns#how-long-do-i-have-to-return-an-item` | 1 | 2 | 1 |
| t2 | Does the return period start from when I ordered or when it was delivered? | `returns#how-long-do-i-have-to-return-an-item` | 1 | 2 | 1 |
| t3 | What conditions does an item need to meet to be returnable? | `returns#which-items-can-be-returned` | 1 | 1 | 1 |
| t4 | Two items from my order came in different parcels. Does each have its own return deadline? | `returns#which-items-can-be-returned` | 1 | 1 | 1 |
| t5 | I just don't need the shoes anymore. Can I swap them for another pair? | `returns#what-can-i-get-for-a-return` | 2 | 2 | 1 |
| t6 | How much will I get back if I return one item from a bigger order? | `returns#how-is-the-refund-worked-out` | 4 | 15 | 9 |
| t7 | My order was cancelled. Can I still return it? | `returns#cancelled-orders` | 1 | 2 | 1 |
| s1 | Is delivery free? | `shipping#how-much-does-shipping-cost` | 2 | 4 | 3 |
| s2 | What's the minimum order to avoid paying a delivery charge? | `shipping#how-much-does-shipping-cost` | 2 | 9 | 4 |
| s3 | Why did I only get half my order in today's package? | `shipping#why-did-my-order-arrive-in-more-than-one-parcel` | 1 | 10 | 4 |
| s4 | My package was due on Monday and still hasn't come. When is it considered lost? | `shipping#my-parcel-is-late-when-does-it-count-as-missing` | 1 | 2 | 1 |
| s5 | My order has no estimated delivery date and it's taking ages. Is it missing? | `shipping#my-parcel-is-late-when-does-it-count-as-missing` | 1 | 1 | 1 |
| s6 | The package never came. Can I get both a replacement and my money back? | `shipping#what-can-i-get-if-my-parcel-is-missing` | 2 | 4 | 2 |
| s7 | Tracking says delivered, but there's nothing at my door. | `shipping#the-carrier-says-it-was-delivered-but-i-didnt-get-it` | 2 | 1 | 1 |
| s8 | The courier couldn't deliver and is sending my parcel back to the warehouse. What happens now? | `shipping#the-parcel-is-being-returned-to-us` | 1 | 3 | 1 |
