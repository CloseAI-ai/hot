interface Env {
  DB: D1Database;
}

export const onRequestGet: PagesFunction<Env> = async (context) => {
  const empty = {
    meta: { progress: 0, current_step: 0, total_steps: 100000, status: "waiting" },
    steps: [], loss: [], loss_ma: [], lr: [], beta: [],
    steps_per_sec: [], tokens_per_sec: [],
  };

  try {
    const row = await context.env.DB.prepare(
      "SELECT * FROM snapshots ORDER BY ts DESC LIMIT 1"
    ).first();

    if (!row) {
      return Response.json(empty, {
        headers: { "Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*" },
      });
    }

    const data = {
      meta: JSON.parse(row.meta as string),
      steps: JSON.parse(row.steps as string),
      loss: JSON.parse(row.loss as string),
      loss_ma: JSON.parse(row.loss_ma as string),
      eval_steps: JSON.parse(row.eval_steps as string),
      eval_loss: JSON.parse(row.eval_loss as string),
      lr: JSON.parse(row.lr as string),
      beta: JSON.parse(row.beta as string),
      steps_per_sec: JSON.parse(row.steps_per_sec as string),
      tokens_per_sec: JSON.parse(row.tokens_per_sec as string),
    };

    return Response.json(data, {
      headers: { "Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*" },
    });
  } catch (e) {
    return Response.json(empty, {
      headers: { "Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*" },
    });
  }
};
