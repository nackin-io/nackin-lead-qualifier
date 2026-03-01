import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import { ChatMessage, LeadData } from '@/types';
import { scoreLead } from '@/lib/scoring';
import { saveLead } from '@/lib/store';

// Lazy-init to avoid build-time instantiation (env vars not available at build)
let _openai: OpenAI | null = null;
function getOpenAI(): OpenAI {
  if (!_openai) {
    _openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return _openai;
}

const SYSTEM_PROMPT = `You are a friendly and professional lead qualification assistant for a web development agency. Your goal is to have a natural conversation with potential clients to understand their needs.

IMPORTANT INSTRUCTIONS:
1. Be conversational, warm, and professional
2. Gradually gather: name, email, company/business, project need/description, budget
3. Don't ask all questions at once - have a natural conversation
4. Extract information naturally from what they say
5. After gathering key information, provide a brief summary and next steps
6. Always be helpful and enthusiastic about their project

EXTRACTION FORMAT - After each response, include a JSON block at the very end wrapped in <<<LEAD_DATA>>> tags:
<<<LEAD_DATA>>>
{
  "name": "extracted name or null",
  "email": "extracted email or null",
  "company": "extracted company/business or null",
  "need": "extracted project need or null",
  "budget": "extracted budget or null",
  "complete": false
}
<<<END_LEAD_DATA>>>

Set "complete": true only when you have gathered at minimum: email + project need + some budget indication, and you've given them next steps.

Start by greeting warmly and asking what they're looking to build.`;

const MOCK_INITIAL_GREETING = 'Hi there! 👋 I am here to help learn about your project. What are you looking to build?';
const MOCK_TOPIC_QUESTIONS = [
  [
    'To help us scope this properly, what company or business is this for?',
    'Quick one so I can tailor recommendations: what company or business is this for?',
  ],
  [
    'What budget range are you aiming for?',
    'Do you have a rough budget range in mind for this project?',
  ],
  [
    'What timeline are you targeting for launch?',
    'When would you ideally like to have this live?',
  ],
  [
    'What is the best email for project follow-up and next steps?',
    'What email should we use to send a proposal and timeline?',
  ],
];
const MOCK_ACKNOWLEDGEMENTS = [
  'Thanks for sharing that.',
  'That helps a lot.',
  'Great context, thank you.',
  'Understood, that makes sense.',
];

interface MockResponse {
  content: string;
  leadData: Partial<LeadData>;
  complete: boolean;
}

function sanitizeExistingLead(existingLead?: Partial<LeadData>): Partial<LeadData> {
  const sanitizedExisting: Partial<LeadData> = {};
  if (!existingLead) return sanitizedExisting;

  if (existingLead.name)    sanitizedExisting.name    = String(existingLead.name).slice(0, 200);
  if (existingLead.email)   sanitizedExisting.email   = String(existingLead.email).slice(0, 254);
  if (existingLead.company) sanitizedExisting.company = String(existingLead.company).slice(0, 200);
  if (existingLead.need)    sanitizedExisting.need    = String(existingLead.need).slice(0, 1000);
  if (existingLead.budget)  sanitizedExisting.budget  = String(existingLead.budget).slice(0, 100);

  return sanitizedExisting;
}

function extractWithPatterns(input: string, patterns: RegExp[]): string | null {
  for (const pattern of patterns) {
    const match = input.match(pattern);
    if (match?.[1]) {
      return match[1].replace(/[.,;!?]+$/g, '').trim().slice(0, 200);
    }
  }
  return null;
}

function extractMockLeadData(text: string, currentLeadData: Partial<LeadData>): Partial<LeadData> {
  const nextLeadData: Partial<LeadData> = { ...currentLeadData };
  const trimmed = text.trim();
  if (!trimmed) return nextLeadData;

  if (!nextLeadData.email) {
    const emailMatch = trimmed.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i);
    if (emailMatch) {
      nextLeadData.email = emailMatch[0].slice(0, 254);
    }
  }

  if (!nextLeadData.company) {
    const company = extractWithPatterns(trimmed, [
      /\b(?:my company is|our company is|our business is|the company is|business is)\s+([A-Za-z0-9][A-Za-z0-9& .,'-]{1,120})/i,
      /\b(?:we are|we're|i'm with|i am with)\s+([A-Za-z0-9][A-Za-z0-9& .,'-]{1,120})/i,
    ]);
    if (company) {
      nextLeadData.company = company;
    }
  }

  if (!nextLeadData.budget) {
    const hasBudgetSignal = /\bbudget|spend|cost|pricing|range|estimate\b/i.test(trimmed)
      || /\$/.test(trimmed)
      || /\b\d+\s?(k|m)\b/i.test(trimmed);
    if (hasBudgetSignal) {
      const budgetMatch = trimmed.match(/\$?\s?\d[\d,.]*\s?(?:k|m)?(?:\s?(?:-|to)\s?\$?\s?\d[\d,.]*\s?(?:k|m)?)?/i);
      if (budgetMatch) {
        nextLeadData.budget = budgetMatch[0].replace(/\s+/g, ' ').trim().slice(0, 100);
      }
    }
  }

  if (!nextLeadData.name) {
    const name = extractWithPatterns(trimmed, [
      /\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z '-]{1,80})/i,
    ]);
    if (name) {
      nextLeadData.name = name;
    }
  }

  if (!nextLeadData.need) {
    nextLeadData.need = trimmed.slice(0, 1000);
  }

  return nextLeadData;
}

function buildMockResponse(messages: ChatMessage[], existingLead: Partial<LeadData>): MockResponse {
  if (messages.length === 0) {
    return {
      content: MOCK_INITIAL_GREETING,
      leadData: { ...existingLead },
      complete: false,
    };
  }

  const userMessages = messages.filter((m) => m.role === 'user');
  const latestUserContent = [...userMessages].reverse().find(Boolean)?.content ?? '';
  const leadData = extractMockLeadData(latestUserContent, existingLead);

  const hasMinimumRequired = Boolean(leadData.email && leadData.need && leadData.budget);
  if (hasMinimumRequired && userMessages.length >= 4) {
    const summaryParts: string[] = [];
    if (leadData.company) summaryParts.push(`company ${leadData.company}`);
    if (leadData.need) summaryParts.push(`project "${leadData.need}"`);
    if (leadData.budget) summaryParts.push(`budget ${leadData.budget}`);
    if (leadData.email) summaryParts.push(`email ${leadData.email}`);
    const summary = summaryParts.join(', ');

    return {
      content: `Perfect, thanks for the details. I captured ${summary}. I will share this with our team and we will follow up with recommended next steps shortly.`,
      leadData,
      complete: true,
    };
  }

  const turnIndex = Math.max(userMessages.length - 1, 0);
  const topicIndex = turnIndex % MOCK_TOPIC_QUESTIONS.length;
  const questionVariants = MOCK_TOPIC_QUESTIONS[topicIndex];
  const question = questionVariants[turnIndex % questionVariants.length];
  const acknowledgement = MOCK_ACKNOWLEDGEMENTS[turnIndex % MOCK_ACKNOWLEDGEMENTS.length];

  return {
    content: `${acknowledgement} ${question}`,
    leadData,
    complete: false,
  };
}

async function finalizeLeadResponse(params: {
  content: string;
  leadData: Partial<LeadData>;
  complete: boolean;
  messages: ChatMessage[];
}) {
  const { content, leadData, complete, messages } = params;

  if (complete && leadData.email) {
    const scoring = scoreLead(leadData);
    const savedLead = saveLead({
      ...leadData,
      score: scoring.score,
      scoreLabel: scoring.label,
      transcript: messages,
      notified: false,
    });

    // Trigger hot lead notification
    if (scoring.label === 'hot') {
      await notifyHotLead(savedLead).catch(console.error);
    }

    return NextResponse.json({
      content,
      leadData,
      scoring,
      leadId: savedLead.id,
      complete: true,
    });
  }

  return NextResponse.json({
    content,
    leadData,
    complete: false,
  });
}

export async function POST(req: NextRequest) {
  try {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { messages, sessionId: _sessionId, existingLead } = await req.json() as {
      messages: ChatMessage[];
      sessionId: string;
      existingLead?: Partial<LeadData>;
    };

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json({ error: 'Invalid messages' }, { status: 400 });
    }

    // Guard against message flooding (cost / DoS protection)
    if (messages.length > 50) {
      return NextResponse.json({ error: 'Too many messages in session' }, { status: 400 });
    }

    const sanitizedExisting = sanitizeExistingLead(existingLead);
    if (!process.env.OPENAI_API_KEY?.trim()) {
      const mock = buildMockResponse(messages, sanitizedExisting);
      return finalizeLeadResponse({
        content: mock.content,
        leadData: mock.leadData,
        complete: mock.complete,
        messages,
      });
    }

    const openaiMessages = [
      { role: 'system' as const, content: SYSTEM_PROMPT },
      ...messages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })).filter(m => m.role === 'user' || m.role === 'assistant'),
    ];

    let rawContent = '';
    try {
      const completion = await getOpenAI().chat.completions.create({
        model: 'gpt-4o',
        messages: openaiMessages,
        temperature: 0.7,
        max_tokens: 500,
      });

      rawContent = completion.choices[0]?.message?.content || '';
    } catch (openaiError) {
      console.error('OpenAI chat completion failed. Falling back to mock response:', openaiError);
      const mock = buildMockResponse(messages, sanitizedExisting);
      return finalizeLeadResponse({
        content: mock.content,
        leadData: mock.leadData,
        complete: mock.complete,
        messages,
      });
    }

    // Extract lead data from response
    const leadData: Partial<LeadData> = { ...sanitizedExisting };
    let isComplete = false;
    
    const leadDataMatch = rawContent.match(/<<<LEAD_DATA>>>([\s\S]*?)<<<END_LEAD_DATA>>>/);
    if (leadDataMatch) {
      try {
        const extracted = JSON.parse(leadDataMatch[1].trim());
        // Merge with existing, only update non-null values
        if (extracted.name) leadData.name = extracted.name;
        if (extracted.email) leadData.email = extracted.email;
        if (extracted.company) leadData.company = extracted.company;
        if (extracted.need) leadData.need = extracted.need;
        if (extracted.budget) leadData.budget = extracted.budget;
        isComplete = extracted.complete === true;
      } catch {
        // JSON parse failed, continue without extracted data
      }
    }

    // Clean response (remove the JSON block from user-visible content)
    const cleanContent = rawContent
      .replace(/<<<LEAD_DATA>>>[\s\S]*?<<<END_LEAD_DATA>>>/g, '')
      .trim();

    return finalizeLeadResponse({
      content: cleanContent,
      leadData,
      complete: isComplete,
      messages,
    });

  } catch (error) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      { error: 'Failed to process chat', details: String(error) },
      { status: 500 }
    );
  }
}

async function notifyHotLead(lead: LeadData) {
  const webhookUrl = process.env.WEBHOOK_URL;
  if (!webhookUrl || webhookUrl.includes('example.com')) {
    // Hot lead detected - webhook notification skipped in demo mode
    return;
  }

  await fetch(webhookUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event: 'hot_lead',
      lead: {
        name: lead.name,
        email: lead.email,
        company: lead.company,
        need: lead.need,
        budget: lead.budget,
        score: lead.score,
      },
      timestamp: new Date().toISOString(),
    }),
  });
}
