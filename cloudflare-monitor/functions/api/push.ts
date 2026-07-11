interface Env {
  DB: D1Database;
  PUSH_SECRET: string;
}

export const onRequestPost: PagesFunction<Env> = async (context) => {
  const auth = context.request.headers.get("Authorization");
  const expected = `Bearer ${context.env.PUSH_SECRET}`;

  if (!auth || auth !== expected) {
    return Response.json({ error: "unauthorized" }, { status: 401 });
  }

  try {
    const data: any = await context.request.json();
    const ts = Math.floor(Date.now() / 1000);

    await context.env.DB.prepare(
      `INSERT INTO snapshots (ts, meta, steps, loss, loss_ma, eval_steps, eval_loss, lr, beta, steps_per_sec, tokens_per_sec)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
      .bind(
        ts,
        JSON.stringify(data.meta),
        JSON.stringify(data.steps),
        JSON.stringify(data.loss),
        JSON.stringify(data.loss_ma),
        JSON.stringify(data.eval_steps || []),
        JSON.stringify(data.eval_loss || []),
        JSON.stringify(data.lr),
        JSON.stringify(data.beta),
        JSON.stringify(data.steps_per_sec),
        JSON.stringify(data.tokens_per_sec)
      )
      .run();

    // 清理旧数据，只保留最近 20 条
    await context.env.DB.prepare(
      "DELETE FROM snapshots WHERE id NOT IN (SELECT id FROM snapshots ORDER BY ts DESC LIMIT 20)"
    ).run();

    return Response.json({ success: true });
  } catch (e: any) {
    return Response.json({ error: e.message }, { status: 500 });
  }
};
