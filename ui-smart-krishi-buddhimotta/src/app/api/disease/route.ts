import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  const baseUrl = process.env.BACKEND_URL?.trim().replace(/\/+$/, "");
  if (!baseUrl) {
    return NextResponse.json({
      error: "ছবি বিশ্লেষণের জন্য Python backend-এর BACKEND_URL সেট করা নেই।",
      code: "BACKEND_NOT_CONFIGURED",
    }, { status: 503 });
  }

  let body: { image_data_url?: unknown; question?: unknown; symptom?: unknown; include_ai_explanation?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "ছবির অনুরোধটি পড়া যায়নি।" }, { status: 400 });
  }
  if (typeof body.image_data_url !== "string" || body.image_data_url.length > 2_900_000) {
    return NextResponse.json({ error: "ছবির আকার কমিয়ে আবার চেষ্টা করুন।" }, { status: 413 });
  }

  try {
    const response = await fetch(`${baseUrl}/api/disease`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data_url: body.image_data_url,
        question: typeof body.question === "string"
          ? body.question.slice(0, 500)
          : typeof body.symptom === "string" ? body.symptom.slice(0, 500) : "",
        include_ai_explanation: body.include_ai_explanation === true,
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(55_000),
    });
    const data = await response.json().catch(() => ({})) as Record<string, unknown>;
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "ছবি বিশ্লেষণ করা যায়নি। পরে আবার চেষ্টা করুন।";
      return NextResponse.json({ error: detail, code: "DISEASE_ANALYSIS_FAILED" }, { status: response.status });
    }
    return NextResponse.json(data, { status: 200 });
  } catch {
    return NextResponse.json({
      error: "Python backend-এ সংযোগ পাওয়া যাচ্ছে না। ফ্রি হোস্টে প্রথম অনুরোধে চালু হতে কিছুটা সময় লাগতে পারে।",
      code: "BACKEND_UNAVAILABLE",
    }, { status: 502 });
  }
}
