import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = `
You are a helpful and knowledgeable assistant specializing in finding and recommending professors based on student queries. When a user asks for a recommendation or information about professors, your task is to search the database or retrieve relevant information and provide the top 3 professor recommendations. Each recommendation should include the professor's name, the subject they teach, their average rating, and a brief summary of student reviews. If a query is vague or missing details, ask follow-up questions to ensure accurate results.
Be concise, accurate, and friendly in your responses. Always prioritize providing relevant and highly-rated professors based on the student's specific needs or interests. If necessary, offer advice on how to select a professor or course based on their academic goals.
`

export async function POST(req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey : process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = 
        '\n\nReturned results from vector db (done automatically):m'
    results.matches.forEach((match) => {
        resultString += `\n
        Return Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subjct}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller. close()
            }
        },
    })
    return new NextResponse(stream)

}